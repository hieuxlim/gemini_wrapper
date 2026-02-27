import asyncio
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemini CLI Execution Node")


class GenerateRequest(BaseModel):
    prompt: str
    structured_output: Optional[str] = None
    model: str = "gemini-2.5-flash"


@app.post("/api/v1/generate")
async def execute_cli(req: GenerateRequest):
    logger.info(f"Received execution request for model: {req.model}")

    try:

        if req.structured_output:
            req.prompt += (
                f"IMPORTANT: You must respond EXACTLY with a valid JSON object that matches the following schema. "
                f"Do NOT wrap the JSON in markdown code blocks (e.g., ```json ... ```) or add any other text.\n"
                f"Schema:\n{req.structured_output}"
            )

        # 1. Execute the CLI command with safety timeout
        process = await asyncio.create_subprocess_exec(
            "gemini",
            "-p",
            req.prompt,
            "--output-format",
            "json",
            "-m",
            req.model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Prevent infinite hangs if the CLI gets stuck
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            logger.error("Gemini CLI execution timed out.")
            raise HTTPException(status_code=504, detail="CLI Execution Timeout")

        if process.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            logger.error(f"Gemini CLI error: {err_msg}")
            raise HTTPException(status_code=500, detail=f"CLI Error: {err_msg}")

        # 2. Parse the CLI wrapper output
        output = stdout.decode("utf-8", errors="replace").strip()
        json_start_idx = output.find("{")
        if json_start_idx == -1:
            raise HTTPException(
                status_code=500, detail="Could not find JSON in CLI output."
            )

        json_str = output[json_start_idx:]
        data = json.loads(json_str)
        raw_response = data.get("response", "").strip()

        if not raw_response:
            raise HTTPException(status_code=500, detail="Empty response from model.")

        # 3. Clean markdown blocks from the raw response
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.startswith("```"):
            raw_response = raw_response[3:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]

        raw_response = raw_response.strip()

        # 4. Return the clean, parsed dictionary
        parsed_content = json.loads(raw_response)
        return {"data": parsed_content}

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model: {e}")
        raise HTTPException(
            status_code=500, detail="Model returned invalid JSON format."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during CLI execution")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

# nyx/tools/computer_use_agent.py

import base64
import time
from openai import OpenAI
from playwright.sync_api import sync_playwright

client = OpenAI()

class ComputerUseAgent:
    def __init__(self, logger=None):
        self.logger = logger

    def get_screenshot(self, page) -> str:
        screenshot = page.screenshot()
        return base64.b64encode(screenshot).decode("utf-8")

    def handle_response_loop(self, page, initial_prompt: str):
        screenshot_base64 = self.get_screenshot(page)

        response = client.responses.create(
            model="computer-use-preview",
            tools=[{
                "type": "computer_use_preview",
                "display_width": 1024,
                "display_height": 768,
                "environment": "browser"
            }],
            input=[{
                "role": "user",
                "content": initial_prompt
            }],
            reasoning={"generate_summary": "detailed"},
            truncation="auto"
        )

        while True:
            computer_calls = [item for item in response.output if item.type == "computer_call"]
            if not computer_calls:
                summary = next((i for i in response.output if i.type == "reasoning"), None)
                return summary.summary[0].text if summary else "No summary."

            call = computer_calls[0]
            action = call.action

            self.execute_action(page, action)
            time.sleep(1)

            screenshot_base64 = self.get_screenshot(page)

            response = client.responses.create(
                model="computer-use-preview",
                previous_response_id=response.id,
                tools=[{
                    "type": "computer_use_preview",
                    "display_width": 1024,
                    "display_height": 768,
                    "environment": "browser"
                }],
                input=[{
                    "call_id": call.call_id,
                    "type": "computer_call_output",
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_base64}"
                    }
                }],
                truncation="auto"
            )

    def execute_action(self, page, action):
        match action["type"]:
            case "click":
                x, y = action["x"], action["y"]
                page.mouse.click(x, y)
            case "scroll":
                scrollX, scrollY = action.get("scrollX", 0), action.get("scrollY", 0)
                page.evaluate(f"window.scrollBy({scrollX}, {scrollY})")
            case "type":
                text = action.get("text", "")
                page.keyboard.type(text)
            case "keypress":
                for k in action.get("keys", []):
                    page.keyboard.press(k)
            case "wait":
                time.sleep(2)

    def run_task(self, url: str, prompt: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-extensions", "--disable-file-system"]
            )
            page = browser.new_page()
            page.set_viewport_size({"width": 1024, "height": 768})
            page.goto(url)
            time.sleep(3)
            result = self.handle_response_loop(page, prompt)
            browser.close()
            return result

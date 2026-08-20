# RoboLab WebUI

This is the B6 React + TypeScript control surface. It consumes the loopback
FastAPI endpoints under `/api/v1`; it does not contain a second execution
path. Build it with `npm install && npm run build`, then start the local
service from the repository root with `robolab serve`.

The production `dist/` output is tracked so a normal RoboLab run does not
require Node.js. The UI keeps the first navigation level limited to Dashboard,
Robots, Skills, Jobs, Artifacts and Settings, with a compact neutral visual
system and no gradients, glow or glass effects.

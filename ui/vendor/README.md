# Browser rendering dependency

Three.js **0.180.0**, including its matching STLLoader and OrbitControls, is
vendored from https://registry.npmjs.org/three/-/three-0.180.0.tgz.
The unmodified upstream modules and MIT license are under `three/`.
Only the needed build and example modules are included; no runtime CDN access
or Node build step is required. Original STL assets are served locally by the
Python backend's allowlisted `/robot-assets/` routes.

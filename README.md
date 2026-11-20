🎨 CHITRA — Image Generation API & Web Application

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=venom&color=0:111827,100:7C3AED&height=200&section=header&text=CHITRA&fontSize=75&fontColor=ffffff&animation=fadeIn&fontAlignY=40&desc=Create.%20Integrate.%20Explore.&descSize=18&descAlignY=65" width="100%" alt="CHITRA Banner"/>
</p><p align="center">
  <b>A lightweight, self-hosted image generation platform built with Python and Flask.</b>
  <br/>
  Generate images through a REST API or an easy-to-use web dashboard.
</p><p align="center">
  <a href="https://github.com/Rg100152/chitra">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-Powered-black?style=for-the-badge&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/API-REST-6C63FF?style=for-the-badge" alt="REST API"/>
  <img src="https://img.shields.io/badge/Status-Open%20Source-brightgreen?style=for-the-badge" alt="Open Source"/>
</p><p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-api-documentation">API Docs</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-contributing">Contributing</a>
</p>---

📖 About CHITRA

CHITRA is a single-file Python application that combines an image generation REST API and a web application in one place.

It uses the official OpenAI Images API to generate images from text prompts and provides local image storage, generation history, authentication options, and request rate limiting.

Whether you're a developer building an application, a student learning Python, or an enthusiast experimenting with APIs, CHITRA provides a simple starting point.

🎯 Who is it for?

- 👨‍💻 Developers building AI-powered applications.
- 🎓 Students learning Python, Flask and REST APIs.
- 🧪 Beginners experimenting with AI integrations.
- 🛠️ Open-source contributors and hobby developers.
- 🌐 Anyone who wants a self-hosted image generation service.

✨ Features

Feature| Description
🖼️ Image Generation| Generate images from text prompts
🔌 REST API| Integrate image generation into other applications
🌐 Web Dashboard| Access the application through a browser
📦 Single File| Main application contained in one Python file
🗂️ Image History| Store generated image metadata locally
🔐 API Authentication| Optional Bearer token authentication
⚡ Rate Limiting| Limit requests per IP address
🎨 Multiple Formats| PNG, JPEG and WebP
📐 Image Sizes| Square, landscape, portrait and auto
🧠 Model Selection| GPT Image and GPT Image Mini
🗑️ Image Management| Retrieve and delete generated images
⚙️ Environment Configuration| Configure the application without editing source code

🛠️ Technology Stack

- Python — Application logic.
- Flask — Web framework and REST API.
- OpenAI Python SDK — Image generation integration.
- JSON — Local metadata management.
- HTML/CSS/JavaScript — Web dashboard.

🚀 Installation

1. Clone the repository

git clone https://github.com/Rg100152/chitra.git

cd chitra

2. Create a virtual environment (recommended)

python -m venv venv

Activate it:

Linux / macOS

source venv/bin/activate

Windows

venv\Scripts\activate

3. Install dependencies

pip install flask openai

4. Configure your OpenAI API key

Linux / macOS

export OPENAI_API_KEY="your_openai_api_key"

Windows PowerShell

$env:OPENAI_API_KEY="your_openai_api_key"

You need an OpenAI API key with access to the relevant image model. API usage may incur charges.

5. Run CHITRA

python chitra.py

Open your browser:

http://127.0.0.1:5050

That's it! Your local CHITRA server is running.

⚙️ Configuration

CHITRA supports environment variables for customization.

Variable| Default| Description
"OPENAI_API_KEY"| None| Required OpenAI API key
"CHITRA_API_KEY"| None| Optional API authentication
"CHITRA_HOST"| "127.0.0.1"| Server host
"CHITRA_PORT"| "5050"| Server port
"CHITRA_OUTPUT_DIR"| "./chitra_output"| Generated image storage
"CHITRA_MODEL"| "gpt-image-1"| Default image model
"CHITRA_RATE_LIMIT"| "10"| Requests per minute per IP

Example configuration

export OPENAI_API_KEY="your_key"
export CHITRA_API_KEY="your_secret_token"
export CHITRA_HOST="127.0.0.1"
export CHITRA_PORT="5050"

python chitra.py

Important: Keep your API keys private. Never commit secrets to GitHub or expose them in frontend JavaScript.

🔌 API Documentation

Base URL:

http://127.0.0.1:5050

Available Endpoints

Method| Endpoint| Description
GET| "/health"| Check server health
GET| "/v1/docs"| View API documentation
GET| "/v1/models"| List supported models
POST| "/v1/images/generations"| Generate images
GET| "/v1/images"| List generated images
GET| "/v1/images/<image_id>"| Retrieve an image
DELETE| "/v1/images/<image_id>"| Delete an image

1. Generate an image

Endpoint

POST /v1/images/generations
Content-Type: application/json

Example request:

{
  "prompt": "A futuristic city at night, cinematic lighting, ultra detailed",
  "model": "gpt-image-1",
  "size": "1024x1024",
  "quality": "high",
  "output_format": "png",
  "background": "auto",
  "n": 1
}

2. cURL Example

curl -X POST http://127.0.0.1:5050/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_secret_token" \
  -d '{
    "prompt": "A beautiful futuristic landscape",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "quality": "high",
    "output_format": "png",
    "n": 1
  }'

If "CHITRA_API_KEY" is not configured, the Authorization header is not required.

3. Python Integration

import requests

API_URL = "http://127.0.0.1:5050"

response = requests.post(
    f"{API_URL}/v1/images/generations",
    json={
        "prompt": "A futuristic AI laboratory",
        "model": "gpt-image-1",
        "size": "1024x1024",
        "quality": "high",
        "output_format": "png",
        "n": 1
    },
    headers={
        "Authorization": "Bearer your_secret_token"
    }
)

print(response.status_code)
print(response.json())

Install the client dependency if needed:

pip install requests

4. List Generated Images

curl http://127.0.0.1:5050/v1/images

5. Check Server Health

curl http://127.0.0.1:5050/health

6. Retrieve an Image

http://127.0.0.1:5050/v1/images/IMAGE_ID

Replace "IMAGE_ID" with the actual ID returned by the API.

🧠 Supported Models

CHITRA currently exposes:

Model| Purpose
"gpt-image-1"| Image generation
"gpt-image-1-mini"| Image generation

Available image sizes:

- "1024x1024"
- "1536x1024"
- "1024x1536"
- "auto"

Available quality settings:

- "low"
- "medium"
- "high"
- "auto"

Supported output formats:

- PNG
- JPEG
- WebP

See the API documentation for the currently implemented capabilities.

📁 Project Structure

chitra/
│
├── chitra.py
├── README.md
│
└── chitra_output/
    ├── _metadata.json
    ├── image-id-1.png
    └── image-id-2.png

The "chitra_output" directory is automatically created when the application starts.

🔒 Security Notes

CHITRA includes basic security features:

- Optional Bearer token authentication.
- Constant-time token comparison.
- Per-IP in-memory rate limiting.
- Image ID and extension validation.
- Restricted image file paths.
- Request size limit.

For production deployments:

- Configure "CHITRA_API_KEY".
- Keep the OpenAI API key server-side.
- Use HTTPS.
- Run behind a properly configured production WSGI server.
- Consider persistent rate limiting and storage.
- Restrict access to image management endpoints.
- Review authentication and deployment settings before exposing the service publicly.

The built-in Flask development server is intended for development, not production.

🌍 Open Source & Contributions

CHITRA is intended to be useful to developers, students and the open-source community.

Contributions, suggestions, bug reports and improvements are welcome!

How to contribute

1. Fork this repository.
2. Create a new branch.
3. Make your improvements.
4. Test your changes.
5. Submit a Pull Request.

You can also contribute by:

- Reporting bugs.
- Suggesting new features.
- Improving documentation.
- Sharing integration examples.
- Helping beginners understand the project.

💡 Ideas for Future Improvements

Some possible directions for community contributions:

- Improved image gallery.
- Better generation history interface.
- Additional API integrations.
- Docker support.
- Database-backed metadata.
- Improved monitoring and logging.
- More authentication options.

These are potential improvements, not currently implemented features.

⚠️ Important Disclaimer

CHITRA is an independent open-source project and is not an official OpenAI product.

Image generation is powered by the OpenAI API. Users are responsible for their own API keys, API usage, applicable charges and compliance with relevant service policies.

📜 License

License status: To be confirmed.

Before redistributing or reusing this project, check the repository's license. A publicly accessible GitHub repository is not automatically licensed for unrestricted reuse.

If you want unrestricted community contributions under a standard permissive license, consider adding an MIT License.

👨‍💻 Author

Raj Gautam

BCA Student | Python Learner | Linux & Open-Source Enthusiast

<p align="center">
  <a href="https://github.com/Rg100152">
    <img src="https://img.shields.io/badge/GitHub-Raj%20Gautam-181717?style=for-the-badge&logo=github" alt="GitHub Profile"/>
  </a>
  <a href="https://github.com/Rg100152/chitra">
    <img src="https://img.shields.io/badge/Explore-CHITRA-7C3AED?style=for-the-badge&logo=github" alt="Explore CHITRA"/>
  </a>
</p>---

<p align="center">
  <b>Built for developers. Useful for students. Open to everyone.</b>
  <br/>
  ⭐ If you find CHITRA useful, consider giving this repository a star!
  <br/><br/>
  <i>Create • Experiment • Learn • Share</i>
</p>

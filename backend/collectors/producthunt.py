import requests
import json
from datetime import datetime

def fetch_producthunt(api_key):
    """
    Product Hunt API에서 AI 툴 데이터를 가져옵니다.
    실제 API 키가 없으므로 더미 데이터를 반환합니다.
    """
    print(f"Product Hunt API로부터 데이터를 가져옵니다. (API Key: {api_key})")
    # 실제 GraphQL 쿼리 및 API 호출은 생략
    # 예시 더미 데이터 반환
    return [
        # AI 전용 툴 (AI & ML, Natural Language Processing, ai)
        {
            "name": "ChatGPT",
            "description": "AI chatbot for natural language understanding and generation.",
            "url": "https://chat.openai.com/",
            "category": "Natural Language Processing",
            "tags": ["nlp", "ai", "chatbot"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Midjourney",
            "description": "AI art generator creating images from text prompts.",
            "url": "https://www.midjourney.com/",
            "category": "Image Generation",
            "tags": ["ai", "art", "image"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "DALL-E 3",
            "description": "Generative AI model by OpenAI for creating images from text.",
            "url": "https://openai.com/dall-e-3",
            "category": "Image Generation",
            "tags": ["ai", "image", "generative"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Google Bard",
            "description": "Conversational AI chatbot by Google.",
            "url": "https://bard.google.com/",
            "category": "Natural Language Processing",
            "tags": ["ai", "nlp", "chatbot"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "RunwayML",
            "description": "AI magic tools for video editing, image generation, and more.",
            "url": "https://runwayml.com/",
            "category": "Video Editing",
            "tags": ["ai", "video", "generative"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "ElevenLabs",
            "description": "AI voice generator for realistic text-to-speech.",
            "url": "https://elevenlabs.io/",
            "category": "Audio",
            "tags": ["ai", "voice", "text-to-speech"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Stable Diffusion",
            "description": "Open-source deep learning model for generating images.",
            "url": "https://stability.ai/stable-diffusion",
            "category": "Image Generation",
            "tags": ["ai", "image", "open-source"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Hugging Face",
            "description": "Platform for machine learning models, datasets, and demos.",
            "url": "https://huggingface.co/",
            "category": "AI & ML",
            "tags": ["ai", "ml", "models"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "LangChain",
            "description": "Framework for developing applications powered by language models.",
            "url": "https://www.langchain.com/",
            "category": "Developer Tools",
            "tags": ["ai", "nlp", "framework"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Whisper (OpenAI)",
            "description": "AI model for robust speech recognition.",
            "url": "https://openai.com/research/whisper",
            "category": "Audio",
            "tags": ["ai", "speech", "transcription"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        # 개발자용 툴 (Developer Tools, Code Generation, coding, developer)
        {
            "name": "GitHub Copilot",
            "description": "AI pair programmer for code suggestions.",
            "url": "https://github.com/features/copilot",
            "category": "Developer Tools",
            "tags": ["coding", "developer", "ai"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "VS Code",
            "description": "Free code editor for web, mobile, and cloud development.",
            "url": "https://code.visualstudio.com/",
            "category": "Developer Tools",
            "tags": ["coding", "editor"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Docker",
            "description": "Platform for developing, shipping, and running applications in containers.",
            "url": "https://www.docker.com/",
            "category": "Developer Tools",
            "tags": ["developer", "devops"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Postman",
            "description": "API platform for building and using APIs.",
            "url": "https://www.postman.com/",
            "category": "Developer Tools",
            "tags": ["api", "testing"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "GitLab",
            "description": "Complete DevOps platform, delivered as a single application.",
            "url": "https://about.gitlab.com/",
            "category": "Developer Tools",
            "tags": ["devops", "git"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Jira",
            "description": "Software development tool for agile teams to plan, track, and release great software.",
            "url": "https://www.atlassian.com/software/jira",
            "category": "Project Management",
            "tags": ["developer", "management"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Swagger UI",
            "description": "Visualize and interact with the API's resources without any of the implementation logic in place.",
            "url": "https://swagger.io/tools/swagger-ui/",
            "category": "Developer Tools",
            "tags": ["api", "documentation"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "PyCharm",
            "description": "Python IDE for professional developers.",
            "url": "https://www.jetbrains.com/pycharm/",
            "category": "Developer Tools",
            "tags": ["coding", "python"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Tableau",
            "description": "Visual analytics platform transforming the way people use data to solve problems.",
            "url": "https://www.tableau.com/",
            "category": "Business Intelligence",
            "tags": ["data", "developer"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Terraform",
            "description": "Infrastructure as Code tool for building, changing, and versioning infrastructure safely and efficiently.",
            "url": "https://www.terraform.io/",
            "category": "DevOps Tools",
            "tags": ["devops", "infrastructure"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        # 일반 사무용 툴 (Productivity, Business, office, productivity)
        {
            "name": "Microsoft Office 365",
            "description": "Suite of productivity tools including Word, Excel, PowerPoint, and Outlook.",
            "url": "https://www.microsoft.com/en-us/microsoft-365",
            "category": "Productivity",
            "tags": ["office", "productivity"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Google Workspace",
            "description": "Cloud-based suite of productivity and collaboration tools.",
            "url": "https://workspace.google.com/",
            "category": "Productivity",
            "tags": ["office", "collaboration"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Zoom",
            "description": "Video conferencing and online meeting platform.",
            "url": "https://zoom.us/",
            "category": "Communication",
            "tags": ["productivity", "meeting"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Slack",
            "description": "Channel-based messaging platform.",
            "url": "https://slack.com/",
            "category": "Communication",
            "tags": ["productivity", "collaboration"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Trello",
            "description": "Visual tool for organizing your work and life.",
            "url": "https://trello.com/",
            "category": "Project Management",
            "tags": ["productivity", "management"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Asana",
            "description": "Work management platform to organize, track, and manage your team's work.",
            "url": "https://asana.com/",
            "category": "Project Management",
            "tags": ["productivity", "management"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Canva",
            "description": "Online graphic design tool for creating social media graphics, presentations, and more.",
            "url": "https://www.canva.com/",
            "category": "Design Tools",
            "tags": ["productivity", "design"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Evernote",
            "description": "Note-taking app for organizing notes, tasks, and archives.",
            "url": "https://evernote.com/",
            "category": "Productivity",
            "tags": ["notes", "organization"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "Grammarly",
            "description": "AI writing assistant to help you write clearly and effectively.",
            "url": "https://www.grammarly.com/",
            "category": "Writing Tools",
            "tags": ["productivity", "writing"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        },
        {
            "name": "DocuSign",
            "description": "Electronic signature and agreement cloud.",
            "url": "https://www.docusign.com/",
            "category": "Business",
            "tags": ["documents", "business"],
            "source": "Product Hunt",
            "collected_at": datetime.now().isoformat() + "Z"
        }
    ]

if __name__ == "__main__":
    dummy_api_key = "YOUR_PRODUCT_HUNT_API_KEY"
    data = fetch_producthunt(dummy_api_key)
    print(f"가져온 데이터 수: {len(data)}")
    print(json.dumps(data, indent=2)) 
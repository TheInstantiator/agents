import os
import aiosmtplib  # New import
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env', override=True)

async def send_email(to: str, subject: str, body: str):
    """
    Send an email asynchronously using Gmail SMTP.
    """
    email_user = os.getenv('EMAIL_USER')
    email_pass = os.getenv('EMAIL_PASS')

    # print (f"EMAIL_USER: {email_user}")
    
    if not email_user or not email_pass:
        raise ValueError("EMAIL_USER and EMAIL_PASS must be set in .env")
    
    # Set up the email message
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = to
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = aiosmtplib.SMTP(hostname='smtp.gmail.com', port=587, start_tls=True)
        await server.connect()
        await server.login(email_user, email_pass)
        await server.sendmail(email_user, to, msg.as_string())
        await server.quit()
        print(f"Email sent successfully to {to}")
    except Exception as e:
        print(f"Error sending email: {e}")
        raise


def create_email_template(content: str, company_name: str = "ComplAI") -> str:
    """
    Create a professional HTML email template with the given content.
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }}
            .container {{
                max-width: 600px;
                margin: 20px auto;
                background-color: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px 20px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .content {{
                padding: 30px 20px;
                background-color: white;
            }}
            .content p {{
                margin: 15px 0;
            }}
            .button {{
                display: inline-block;
                padding: 12px 30px;
                background-color: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                font-size: 12px;
                color: #666;
                background-color: #f9f9f9;
                border-top: 1px solid #e0e0e0;
            }}
            .footer a {{
                color: #667eea;
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{company_name}</h1>
                <p style="margin: 5px 0 0 0;">AI-Powered Compliance Solutions</p>
            </div>
            <div class="content">
                {content}
            </div>
            <div class="footer">
                <p>&copy; 2024 {company_name}. All rights reserved.</p>
                <p>
                    <a href="mailto:contact@complai.com">Contact Us</a> |
                    <a href="#">Unsubscribe</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """


async def send_html_email(to: str, subject: str, html_body: str):
    """
    Send an HTML email asynchronously using Gmail SMTP.
    """
    email_user = os.getenv('EMAIL_USER')
    email_pass = os.getenv('EMAIL_PASS')

    if not email_user or not email_pass:
        raise ValueError("EMAIL_USER and EMAIL_PASS must be set in .env")

    # Set up the email message
    msg = MIMEMultipart('alternative')
    msg['From'] = email_user
    msg['To'] = to
    msg['Subject'] = subject

    # Attach HTML content
    html_part = MIMEText(html_body, 'html')
    msg.attach(html_part)

    try:
        server = aiosmtplib.SMTP(hostname='smtp.gmail.com', port=587, start_tls=True)
        await server.connect()
        await server.login(email_user, email_pass)
        await server.sendmail(email_user, to, msg.as_string())
        await server.quit()
        print(f"HTML email sent successfully to {to}")
    except Exception as e:
        print(f"Error sending HTML email: {e}")
        raise
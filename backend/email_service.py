import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"



def send_email(receiver_email, subject, message):

    try:

        msg = MIMEMultipart()

        msg["From"] = SENDER_EMAIL
        msg["To"] = receiver_email
        msg["Subject"] = subject

        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)

        server.starttls()

        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        server.sendmail(
            SENDER_EMAIL,
            receiver_email,
            msg.as_string()
        )

        server.quit()

        print("Email sent successfully")

    except Exception as e:

        print("Email error:", e)


def send_booking_confirmation(email, name, date, hour, wait_time):

    subject = "Appointment Confirmation"

    message = f"""

Hello {name},

Your appointment has been successfully booked.

Appointment Details:

Date: {date}
Hour: {hour}:00

Estimated Wait Time: {round(wait_time,2)} minutes

Please arrive 10 minutes early.

Smart Queue System

"""

    send_email(email, subject, message)



def send_reminder(email, name, hour):

    subject = "Appointment Reminder"

    message = f"""

Hello {name},

Reminder: Your appointment is scheduled in 10 minutes.

Appointment Time: {hour}:00

Please arrive at the clinic.

Smart Queue System

"""

    send_email(email, subject, message)
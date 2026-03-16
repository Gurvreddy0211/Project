from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pymysql
import pandas as pd
import numpy as np
import joblib

from backend.explainability import explain_no_show, explain_wait_time, generate_human_explanation
from backend.lime_explainer import explain_lime
from backend.email_service import send_booking_confirmation

from datetime import datetime, timedelta


no_show_model = joblib.load("models/random_forest_no_show_model.pkl")
wait_model = joblib.load("models/wait_time_model.pkl")

print("Models Loaded Successfully")


router = APIRouter()

templates = Jinja2Templates(directory="frontend/templates")



def get_connection():

    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="smart_queue",
        cursorclass=pymysql.cursors.DictCursor
    )



class Appointment(BaseModel):

    user_id: int
    name: str
    phone: str
    appointment_date: str

    hour: int

    distance_km: float
    reported_urgency: str


@router.get("/login.html", response_class=HTMLResponse)
async def login_page(request: Request):

    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):

    return templates.TemplateResponse(
        "register.html",
        {"request": request}
    )


@router.get("/user_dashboard.html", response_class=HTMLResponse)
async def user_dashboard(request: Request):

    return templates.TemplateResponse(
        "user_dashboard.html",
        {"request": request}
    )


@router.get("/admin_dashboard.html", response_class=HTMLResponse)
async def admin_dashboard(request: Request):

    return templates.TemplateResponse(
        "admin_dashboard.html",
        {"request": request}
    )


@router.get("/book_appointment.html", response_class=HTMLResponse)
async def book_appointment(request: Request):

    return templates.TemplateResponse(
        "book_appointment.html",
        {"request": request}
    )

def get_least_busy_counter(cursor):

    cursor.execute("""
    SELECT c.id, COUNT(a.id) as load_count
    FROM counters c
    LEFT JOIN appointments a
        ON c.id = a.counter_id
        AND a.status='pending'
    WHERE c.status='active'
    GROUP BY c.id
    ORDER BY load_count ASC
    LIMIT 1
    """)

    counter = cursor.fetchone()

    return counter["id"] if counter else None

@router.post("/predict")
def predict_queue(data: Appointment):

    data = data.dict()

    user_id = data["user_id"]
    name = data["name"]
    phone = data["phone"]
    appointment_date = data["appointment_date"]

    hour = data["hour"]
    distance = data["distance_km"]
    urgency_str = data["reported_urgency"]


    date_obj = datetime.strptime(appointment_date, "%Y-%m-%d")

    day = date_obj.weekday()
    month = date_obj.month

    is_weekend = 1 if day in [5,6] else 0

      
    conn = get_connection()
    cursor = conn.cursor()

    # Remove patients whose appointment slot passed 1 hour ago
    cursor.execute("""
    UPDATE appointments
    SET status='expired'
    WHERE status='pending'
    AND TIMESTAMP(appointment_date, SEC_TO_TIME(hour*3600))
    < NOW() - INTERVAL 1 HOUR
    """)
    conn.commit()

    cursor.execute("""
    SELECT COUNT(*) as q
    FROM appointments
    WHERE appointment_date=%s
    AND hour=%s
    AND status='pending'
    AND TIMESTAMP(appointment_date, SEC_TO_TIME(hour*3600))
        >= NOW() - INTERVAL 1 HOUR
    """,(appointment_date,hour))

    queue = cursor.fetchone()["q"]

    
    queue_position = queue + 1

    token_number = queue + 1
    patients_ahead = queue

    counter_id = get_least_busy_counter(cursor)

    staff = 5


    booking_lead_hours = np.random.randint(6,48)
    arrival_delay = 0

    queue_pressure = queue / staff
    queue_per_staff = queue / staff

    avg_duration = 25
    backlog = queue * avg_duration
    queue_service_capacity = backlog / staff

    urgency_map = {"low":0,"medium":1,"high":2}
    urgency = urgency_map.get(urgency_str,0)

    hour_sin = np.sin(2*np.pi*hour/24)
    hour_cos = np.cos(2*np.pi*hour/24)

    day_sin = np.sin(2*np.pi*day/7)
    day_cos = np.cos(2*np.pi*day/7)

    df = pd.DataFrame([{

        "hour":hour,
        "day_of_week":day,
        "month":month,
        "is_weekend":is_weekend,

        "booking_lead_hours":booking_lead_hours,
        "arrival_delay":arrival_delay,

        "queue_length_at_arrival":queue,
        "staff_on_duty_at_arrival":staff,

        "queue_pressure":queue_pressure,
        "queue_per_staff":queue_per_staff,
        "estimated_backlog_minutes":backlog,

        "previous_appointments":0,
        "previous_no_shows":0,
        "no_show_rate":0,

        "service_id":1,
        "location_id":1,
        "booking_channel":0,
        "age_band":1,
        "user_type":0,

        "distance_km":distance,
        "reported_urgency":urgency,

        "hour_sin":hour_sin,
        "hour_cos":hour_cos,
        "day_sin":day_sin,
        "day_cos":day_cos
    }])

    df = df[no_show_model.feature_names_in_]

      

    no_show_prob = no_show_model.predict_proba(df)[0][1]

    no_show_prediction = 1 if no_show_prob > 0.4 else 0

      
    df["queue_service_capacity"] = queue_service_capacity
    df["avg_duration_min"] = avg_duration
    df["duration_std_min"] = 5
    df["service_complexity"] = avg_duration / 60

    wait_df = df[wait_model.feature_names_in_]

    wait_time = wait_model.predict(wait_df)[0]

      

    no_show_explain = explain_no_show(df)
    wait_explain = explain_wait_time(wait_df)

    human_text = generate_human_explanation(wait_time, wait_explain)

    lime_df = df[no_show_model.feature_names_in_]
    lime_explanation = explain_lime(lime_df)

    cursor.execute(
        "SELECT email FROM users WHERE id=%s",
        (user_id,)
    )

    user = cursor.fetchone()
    email = user["email"]

      

    cursor.execute("""

    INSERT INTO appointments
    (user_id,name,email,phone,appointment_date,
    hour,token_number,patients_ahead,counter_id,
    day_of_week,month,is_weekend,
    booking_lead_hours,arrival_delay,
    queue_length,staff,distance_km,urgency,
    no_show_prob,no_show_prediction,wait_time)

    VALUES (%s,%s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,
            %s,%s,%s,%s,
            %s,%s,%s)

    """,(user_id,name,email,phone,appointment_date,
        hour,token_number,patients_ahead,counter_id,
        day,month,is_weekend,
        booking_lead_hours,arrival_delay,
        queue,staff,distance,urgency_str,
        float(no_show_prob),no_show_prediction,float(wait_time)))
    
    conn.commit()

    cursor.close()
    conn.close()

      

    try:
        send_booking_confirmation(
            email,
            name,
            appointment_date,
            hour,
            wait_time
        )
    except Exception as e:
        print("Email error:", e)

      
    return {

        "counter_id": counter_id,

        "token_number": token_number,
        "patients_ahead": patients_ahead,

        "queue_position": queue_position,

        "no_show_probability": round(float(no_show_prob),3),

        "estimated_wait_time_minutes": round(float(wait_time),2),

        "no_show_explanation": no_show_explain,

        "wait_time_explanation": wait_explain,

        "lime_explanation": lime_explanation,

        "human_explanation": human_text

    }

@router.get("/appointments")
def get_appointments():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT a.*,
    (
        SELECT COUNT(*)
        FROM appointments b
        WHERE b.appointment_date = a.appointment_date
        AND b.hour = a.hour
        AND b.status = 'pending'
        AND b.token_number < a.token_number
    ) AS patients_ahead_dynamic
    FROM appointments a
    WHERE a.status != 'expired'
    ORDER BY a.appointment_date, a.hour, a.token_number
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data

@router.get("/my_appointments")

def my_appointments(user_id:int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT a.*,
    (
    SELECT COUNT(*)
    FROM appointments b
    WHERE b.appointment_date = a.appointment_date
    AND b.hour = a.hour
    AND b.token_number < a.token_number
    AND b.status='pending'
    ) AS patients_ahead_dynamic
    FROM appointments a
    WHERE a.user_id=%s
    ORDER BY a.appointment_date, a.hour, a.token_number
    """,(user_id,))

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data


  
@router.post("/update_status")

def update_status(id:int,status:str):

    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute(
        "UPDATE appointments SET status=%s WHERE id=%s",
        (status,id)
    )

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"updated"}

@router.post("/reschedule")
def reschedule(id:int,new_date:str,new_hour:int):

    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute("""
    SELECT COUNT(*) as q
    FROM appointments
    WHERE appointment_date=%s
    AND hour=%s
    AND status='pending'
    AND id != %s
    """,(new_date,new_hour,id))

    queue=cursor.fetchone()["q"]

    token=queue+1

    cursor.execute("""
    UPDATE appointments
    SET appointment_date=%s,
        hour=%s,
        token_number=%s,
        status='pending'
    WHERE id=%s
    """,(new_date,new_hour,token,id))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"Appointment rescheduled","new_token":token}

@router.get("/counters")
def get_counters():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM counters ORDER BY counter_number")

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data

@router.post("/update_counter_status")
def update_counter_status(counter_id:int, status:str):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE counters
    SET status=%s
    WHERE id=%s
    """,(status,counter_id))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"counter status updated"}

@router.post("/update_counter_token")
def update_counter_token(counter_id:int, token:int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE counters
    SET current_token=%s
    WHERE id=%s
    """,(token,counter_id))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"counter token updated"}

@router.post("/update_appointment_counter")
def update_appointment_counter(id:int,counter_id:int):

    conn=get_connection()
    cursor=conn.cursor()

    cursor.execute(
        "UPDATE appointments SET counter_id=%s WHERE id=%s",
        (counter_id,id)
    )

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"counter updated"}

@router.post("/toggle_counter")
def toggle_counter(counter_id:int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT status
    FROM counters
    WHERE id=%s
    """,(counter_id,))

    counter = cursor.fetchone()

    if not counter:
        return {"error":"Counter not found"}

    new_status = "inactive" if counter["status"]=="active" else "active"

    cursor.execute("""
    UPDATE counters
    SET status=%s
    WHERE id=%s
    """,(new_status,counter_id))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"Counter updated","status":new_status}

@router.get("/now_serving")
def now_serving():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT counter_id, MIN(token_number) as token
    FROM appointments
    WHERE status='pending'
    GROUP BY counter_id
    """)

    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return data

@router.post("/add_counter")
def add_counter(counter_number:int):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO counters(counter_number,status)
    VALUES (%s,'active')
    """,(counter_number,))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message":"Counter added successfully"}

@router.get("/health")

def health():
    return {"status": "AI Queue System Running"}
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pymysql
from passlib.context import CryptContext

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



def get_connection():

    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="smart_queue",
        cursorclass=pymysql.cursors.DictCursor
    )


class RegisterUser(BaseModel):

    name: str
    email: str
    password: str
    role: str = "user"


class LoginUser(BaseModel):

    email: str
    password: str


  
# PASSWORD FUNCTIONS
  

def hash_password(password: str):

    if len(password) > 72:
        password = password[:72]

    return pwd_context.hash(password)


def verify_password(plain, hashed):

    if len(plain) > 72:
        plain = plain[:72]

    return pwd_context.verify(plain, hashed)



@router.post("/register")

def register_user(user: RegisterUser):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (user.email,)
    )

    existing = cursor.fetchone()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    hashed_password = hash_password(user.password)

    cursor.execute("""

    INSERT INTO users (name,email,password,role)

    VALUES (%s,%s,%s,%s)

    """, (user.name, user.email, hashed_password, user.role))

    conn.commit()

    cursor.close()
    conn.close()

    return {"message": "User registered successfully"}


  
@router.post("/login")

def login_user(user: LoginUser):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (user.email,)
    )

    db_user = cursor.fetchone()

    cursor.close()
    conn.close()

    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {

        "message": "Login successful",

        "user": {

            "id": db_user["id"],
            "name": db_user["name"],
            "email": db_user["email"],
            "role": db_user["role"]

        }

    }
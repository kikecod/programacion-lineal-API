from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from typing import List, Optional
import pulp
from passlib.context import CryptContext
from database import get_db, User, hash_password, verify_password
import json
from json import dumps
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


# Import database models and configuration
from database import User, ProblemSolution, get_db, create_tables

# Create tables on startup
create_tables()

# Initialize password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Helper functions for password hashing
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Pydantic Models for Request Validation
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

    class Config:
        schema_extra = {
            "example": {
                "email": "usuario@example.com",
                "password": "UnaContraseñaSegura123"
            }
        }

class Restriccion(BaseModel):
    coeficientes: List[float]
    relacion: str  # '<=', '>=', '='
    valorDerecho: float

class ProblemaConfig(BaseModel):
    numVariables: int
    numRestricciones: int
    tipo: str  # 'maximizar' o 'minimizar'

class DatosProblema(BaseModel):
    funcionObjetivo: List[float]
    restricciones: List[Restriccion]

class Problema(BaseModel):
    config: ProblemaConfig
    datos: DatosProblema

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserInDB(UserCreate):
    id: int
    password_hash: str

SECRET_KEY = "your-secret-key"  # Reemplaza esto con una clave secreta segura
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="No se pudo validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email ya registrado.")
    
    # Guardar el hash de la contraseña en la columna 'password_hash'
    new_user = User(email=user.email, password_hash=hash_password(user.password))
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"id": new_user.id, "email": new_user.email}

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Credenciales inválidas.")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint para resolver el problema
@app.post("/resolver")
async def resolver_problema(problema: Problema):
    try:
        # Crear el problema de programación lineal
        if problema.config.tipo == "maximizar":
            prob = pulp.LpProblem("ProgramacionLineal", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("ProgramacionLineal", pulp.LpMinimize)

        # Crear variables de decisión
        vars = [pulp.LpVariable(f"X{i+1}", lowBound=0) for i in range(problema.config.numVariables)]

        # Función objetivo
        prob += pulp.lpSum([problema.datos.funcionObjetivo[i] * vars[i] for i in range(len(vars))])

        # Restricciones
        for i, rest in enumerate(problema.datos.restricciones):
            expr = pulp.lpSum([rest.coeficientes[j] * vars[j] for j in range(len(vars))])
            if rest.relacion == "<=":
                prob += expr <= rest.valorDerecho
            elif rest.relacion == ">=":
                prob += expr >= rest.valorDerecho
            elif rest.relacion == "=":
                prob += expr == rest.valorDerecho
            else:
                raise HTTPException(status_code=400, detail=f"Relación desconocida: {rest.relacion}")

        # Resolver el problema usando el método Simplex
        solucionador = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solucionador)

        # Validar que el problema tenga solución
        if pulp.LpStatus[prob.status] != "Optimal":
            raise HTTPException(status_code=400, detail="No se encontró solución óptima")

        # Preparar resultados (rangos simulados)
        rangos_coeficientes = [
            {
                "variable": f"X{i+1}",
                "limiteInferior": 0,  # Simulado
                "valorActual": problema.datos.funcionObjetivo[i],
                "limiteSuperior": "Sin límite"  # Simulado
            }
            for i in range(problema.config.numVariables)
        ]

        rangos_lado_derecho = [
            {
                "restriccion": i+1,
                "limiteInferior": 0,  # Simulado
                "valorActual": rest.valorDerecho,
                "limiteSuperior": "Sin límite"  # Simulado
            }
            for i, rest in enumerate(problema.datos.restricciones)
        ]

        # Preparar resultados
        resultado = {
            "valorFuncionObjetivo": pulp.value(prob.objective),
            "variables": [
                {
                    "nombre": f"X{i+1}",
                    "valor": pulp.value(var),
                    "costoReducido": var.dj if var.dj is not None else 0.0,
                }
                for i, var in enumerate(vars)
            ],
            "restricciones": [
                {
                    "numero": i+1,
                    "holgura": c.slack if c.slack is not None else 0.0,
                    "precioSombra": c.pi if c.pi is not None else 0.0,
                }
                for i, c in enumerate(prob.constraints.values())
            ],
            "rangosCoeficientes": rangos_coeficientes,
            "rangosLadoDerecho": rangos_lado_derecho
        }

        return resultado

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/guardar_ejercicio")
async def guardar_ejercicio(
    ejercicio: dict,  # El JSON completo con los datos del problema y solución
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        # Extraer los datos del JSON
        config = ejercicio.get("config", {})
        datos = ejercicio.get("datos", {})
        resultado = ejercicio.get("resultado", {})

        # Validar que existan los campos necesarios
        if not config or not datos or not resultado:
            raise HTTPException(status_code=400, detail="Datos incompletos en la solicitud.")

        # Crear un registro en la tabla
        new_solution = ProblemSolution(
            user_id=current_user.id,
            problem_type=config.get("tipo"),
            function_objective=dumps(datos.get("funcionObjetivo", [])),  # Serialización JSON
            restrictions=dumps(datos.get("restricciones", [])),  # Serialización JSON
            solution_value=resultado.get("valorFuncionObjetivo"),
            solution_variables=dumps(resultado.get("variables", [])),  # Serialización JSON
            constraints=dumps(resultado.get("restricciones", [])),  # Serialización JSON
            coefficient_ranges=dumps(resultado.get("rangosCoeficientes", [])),  # Serialización JSON
            rhs_ranges=dumps(resultado.get("rangosLadoDerecho", [])),  # Serialización JSON
        )
        
        # Guardar en la base de datos
        db.add(new_solution)
        db.commit()
        db.refresh(new_solution)

        return {"message": "Ejercicio guardado exitosamente", "id": new_solution.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/user/history")
async def get_user_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    exercises = db.query(ProblemSolution).filter(ProblemSolution.user_id == current_user.id).all()
    history = [
        {
            "id": exercise.id,
            "tipoProblema": exercise.problem_type,
            "funcionObjetivo": exercise.function_objective,
            "restricciones": exercise.restrictions,
            "valorSolucion": exercise.solution_value,
            "variablesSolucion": exercise.solution_variables,
            "restriccionesDetalles": exercise.constraints,
            "rangosCoeficientes": exercise.coefficient_ranges,
            "rangosLadoDerecho": exercise.rhs_ranges,
            "resueltoEl": exercise.solved_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        for exercise in exercises
    ]

    return {
        "userId": current_user.id,
        "email": current_user.email,
        "history": history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
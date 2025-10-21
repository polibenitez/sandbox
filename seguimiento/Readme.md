# 🚀 Sistema de Control de Inversiones

Sistema completo para gestionar inversiones con frontend moderno y backend robusto.

## 📁 Estructura del Proyecto

```
investment-tracker/
├── backend/
│   ├── app.py                 # API Flask
│   ├── requirements.txt       # Dependencias Python
│   └── inversiones.db        # Base de datos SQLite (se crea automáticamente)
├── frontend/
│   └── index.html            # Aplicación web
└── README.md                 # Este archivo
```

## ⚙️ Instalación

### 1. Backend (Python/Flask)

```bash
# Crear directorio del proyecto
mkdir investment-tracker
cd investment-tracker
mkdir backend frontend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install flask flask-sqlalchemy flask-cors
```

**Crear `backend/requirements.txt`:**
```
Flask==2.3.3
Flask-SQLAlchemy==3.0.5
Flask-CORS==4.0.0
```

### 2. Frontend

- Copiar el código HTML en `frontend/index.html`
- Abrir directamente en el navegador o usar un servidor local

### 3. Ejecución

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```
El servidor estará disponible en: `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
# Opción 1: Abrir directamente index.html en el navegador
# Opción 2: Usar servidor Python simple
python -m http.server 8080
```
El frontend estará disponible en: `http://localhost:8080`

## 🎯 Características Principales

### Backend (API REST)
- **Base de datos SQLite**: Almacenamiento persistente
- **CRUD completo**: Crear, leer, actualizar, eliminar transacciones
- **Cálculos automáticos**: Posición actual, beneficios, comisiones
- **Endpoints disponibles**:
  - `GET /api/transacciones` - Listar todas las transacciones
  - `POST /api/transacciones` - Crear nueva transacción
  - `DELETE /api/transacciones/<id>` - Eliminar transacción
  - `GET /api/resumen` - Obtener resumen de inversiones
  - `GET /api/estadisticas` - Datos para visualizaciones

### Frontend (Interfaz Web)
- **Dashboard interactivo** con métricas en tiempo real
- **Visualizaciones**:
  - Evolución temporal de la inversión
  - Distribución por valores/acciones
- **Gestión de transacciones** simplificada (sin cantidad exacta)
- **Exportación a Excel** con múltiples hojas
- **Diseño responsive** para móviles y tablets

## 📊 Modelo de Datos

### Tabla: `transacciones`
```sql
CREATE TABLE transaccion (
    id INTEGER PRIMARY KEY,
    fecha DATE NOT NULL,
    valor VARCHAR(100) NOT NULL,     -- Nombre de la acción/valor
    tipo VARCHAR(10) NOT NULL,       -- 'Compra' o 'Venta'
    valor_operacion FLOAT NOT NULL,  -- Valor total de la operación
    comisiones FLOAT DEFAULT 0,      -- Comisiones de la operación
    valor_final FLOAT NOT NULL,      -- valor_operacion +/- comisiones
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 💡 Conceptos Clave

### Diferencias con la Versión Anterior:
1. **Sin cantidad exacta**: Solo valor de la operación (€1000 en lugar de 50 acciones × €20)
2. **Arquitectura separada**: Frontend y backend independientes
3. **Base de datos real**: SQLite para persistencia
4. **API REST**: Comunicación estándar entre frontend y backend
5. **Visualizaciones avanzadas**: Gráficos con Chart.js

### Cálculos Principales:
- **Posición Actual**: Total invertido - Total vendido
- **Beneficio Realizado**: Total vendido - Total invertido (solo operaciones cerradas)
- **Valor Final**: Valor operación ± comisiones (suma en compras, resta en ventas)

## 🛠️ Personalización

### Cambiar Base de Datos
Para usar PostgreSQL o MySQL, modificar en `app.py`:
```python
# PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/inversiones'

# MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://user:password@localhost/inversiones'
```

### Añadir Nuevos Campos
1. Modificar el modelo en `app.py`
2. Actualizar formulario en el frontend
3. Ajustar endpoints de la API

### Configurar CORS para Producción
```python
from flask_cors import CORS
CORS(app, origins=['https://mi-dominio.com'])
```

## 🚀 Despliegue en Producción

### Backend
- **Heroku**: `gunicorn app:app`
- **VPS**: Nginx + Gunicorn
- **Docker**: Containerizar la aplicación

### Frontend
- **Netlify/Vercel**: Deploy estático
- **GitHub Pages**: Hosting gratuito
- **CDN**: Para mejor rendimiento

### Base de Datos
- **PostgreSQL**: Para aplicaciones en producción
- **SQLite**: Suficiente para uso personal

## 📈 Próximas Mejoras

1. **Autenticación de usuarios**
2. **Cotizaciones en tiempo real** (APIs financieras)
3. **Alertas y notificaciones**
4. **Análisis avanzado** (rentabilidad, riesgo)
5. **Importación de datos** desde brokers
6. **Aplicación móvil** (React Native/Flutter)

## 🐛 Solución de Problemas

### Error de CORS
- Verificar que Flask-CORS esté instalado
- Comprobar que el frontend apunte a la URL correcta del backend

### Base de datos no se crea
- Verificar permisos de escritura en el directorio
- Ejecutar `python -c "from app import db; db.create_all()"`

### Gráficos no se muestran
- Verificar conexión a CDN de Chart.js
- Comprobar datos en la consola del navegador

¡El sistema está listo para gestionar tus inversiones de forma profesional! 🎉
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Permitir requests desde el frontend

# Configuración de la base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///inversiones.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# Modelo de la tabla de transacciones
class Transaccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.Date, nullable=False)
    valor = db.Column(db.String(100), nullable=False)  # Nombre de la acción/valor
    tipo = db.Column(db.String(10), nullable=False)  # 'Compra' o 'Venta'
    valor_operacion = db.Column(db.Float, nullable=False)  # Valor total de la operación
    comisiones = db.Column(db.Float, default=0)
    valor_final = db.Column(db.Float, nullable=False)  # valor_operacion +/- comisiones
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "fecha": self.fecha.isoformat(),
            "valor": self.valor,
            "tipo": self.tipo,
            "valor_operacion": self.valor_operacion,
            "comisiones": self.comisiones,
            "valor_final": self.valor_final,
            "created_at": self.created_at.isoformat(),
        }


# Crear las tablas
with app.app_context():
    db.create_all()


@app.route("/api/transacciones", methods=["GET"])
def get_transacciones():
    """Obtener todas las transacciones"""
    transacciones = Transaccion.query.order_by(Transaccion.fecha.desc()).all()
    return jsonify([t.to_dict() for t in transacciones])


@app.route("/api/transacciones", methods=["POST"])
def crear_transaccion():
    """Crear una nueva transacción"""
    try:
        data = request.get_json()

        # Validar datos requeridos
        required_fields = ["fecha", "valor", "tipo", "valor_operacion"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo requerido: {field}"}), 400

        # Calcular valor final considerando comisiones
        comisiones = float(data.get("comisiones", 0))
        valor_operacion = float(data["valor_operacion"])

        if data["tipo"] == "Compra":
            valor_final = valor_operacion + comisiones
        else:  # Venta
            valor_final = valor_operacion - comisiones

        transaccion = Transaccion(
            fecha=datetime.strptime(data["fecha"], "%Y-%m-%d").date(),
            valor=data["valor"],
            tipo=data["tipo"],
            valor_operacion=valor_operacion,
            comisiones=comisiones,
            valor_final=valor_final,
        )

        db.session.add(transaccion)
        db.session.commit()

        return jsonify(transaccion.to_dict()), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/transacciones/<int:transaccion_id>", methods=["DELETE"])
def eliminar_transaccion(transaccion_id):
    """Eliminar una transacción"""
    transaccion = Transaccion.query.get_or_404(transaccion_id)
    db.session.delete(transaccion)
    db.session.commit()
    return jsonify({"message": "Transacción eliminada correctamente"})


@app.route("/api/resumen", methods=["GET"])
def get_resumen():
    """Obtener resumen de inversiones"""
    transacciones = Transaccion.query.all()

    # Calcular métricas generales
    total_invertido = sum(t.valor_final for t in transacciones if t.tipo == "Compra")
    total_vendido = sum(t.valor_final for t in transacciones if t.tipo == "Venta")

    # Posición actual (lo que tienes invertido menos lo vendido)
    posicion_actual = total_invertido - total_vendido

    # Beneficio realizado (solo de las ventas menos las compras correspondientes)
    beneficio_realizado = total_vendido - sum(
        t.valor_final
        for t in transacciones
        if t.tipo == "Compra"
        and any(v.valor == t.valor and v.tipo == "Venta" for v in transacciones)
    )

    # Resumen por valor/acción
    valores_resumen = {}
    for transaccion in transacciones:
        valor = transaccion.valor
        if valor not in valores_resumen:
            valores_resumen[valor] = {
                "valor": valor,
                "total_compras": 0,
                "total_ventas": 0,
                "posicion_actual": 0,
                "beneficio_realizado": 0,
                "num_operaciones": 0,
            }

        resumen = valores_resumen[valor]
        resumen["num_operaciones"] += 1

        if transaccion.tipo == "Compra":
            resumen["total_compras"] += transaccion.valor_final
            resumen["posicion_actual"] += transaccion.valor_final
        else:  # Venta
            resumen["total_ventas"] += transaccion.valor_final
            resumen[
                "posicion_actual"
            ] -= (
                transaccion.valor_operacion
            )  # Restamos el valor de venta de la posición
            resumen["beneficio_realizado"] += transaccion.valor_final

    # Calcular beneficio realizado por valor
    for valor_data in valores_resumen.values():
        if valor_data["total_ventas"] > 0:
            valor_data["beneficio_realizado"] = (
                valor_data["total_ventas"] - valor_data["total_compras"]
            )
        else:
            valor_data["beneficio_realizado"] = 0

    return jsonify(
        {
            "resumen_general": {
                "total_invertido": round(total_invertido, 2),
                "total_vendido": round(total_vendido, 2),
                "posicion_actual": round(posicion_actual, 2),
                "beneficio_realizado": (
                    round(total_vendido - total_invertido, 2)
                    if total_vendido > 0
                    else 0
                ),
                "num_valores": len(valores_resumen),
                "num_transacciones": len(transacciones),
            },
            "por_valor": list(valores_resumen.values()),
        }
    )


@app.route("/api/estadisticas", methods=["GET"])
def get_estadisticas():
    """Obtener estadísticas para visualización"""
    transacciones = Transaccion.query.order_by(Transaccion.fecha).all()

    # Evolución temporal de la inversión
    evolucion = []
    inversion_acumulada = 0

    for transaccion in transacciones:
        if transaccion.tipo == "Compra":
            inversion_acumulada += transaccion.valor_final
        else:
            inversion_acumulada -= transaccion.valor_operacion

        evolucion.append(
            {
                "fecha": transaccion.fecha.isoformat(),
                "inversion_acumulada": round(inversion_acumulada, 2),
                "operacion": f"{transaccion.tipo}: {transaccion.valor} (€{transaccion.valor_final})",
            }
        )

    # Top valores por inversión
    valores_inversion = {}
    for transaccion in transacciones:
        if transaccion.valor not in valores_inversion:
            valores_inversion[transaccion.valor] = 0

        if transaccion.tipo == "Compra":
            valores_inversion[transaccion.valor] += transaccion.valor_final
        else:
            valores_inversion[transaccion.valor] -= transaccion.valor_operacion

    # Filtrar valores con posición positiva y ordenar
    top_valores = [
        {"valor": valor, "inversion": round(inversion, 2)}
        for valor, inversion in valores_inversion.items()
        if inversion > 0
    ]
    top_valores.sort(key=lambda x: x["inversion"], reverse=True)

    return jsonify(
        {
            "evolucion_temporal": evolucion,
            "top_valores": top_valores[:10],  # Top 10
            "distribucion_tipos": {
                "compras": len([t for t in transacciones if t.tipo == "Compra"]),
                "ventas": len([t for t in transacciones if t.tipo == "Venta"]),
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

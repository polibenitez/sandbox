from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import os
import pandas as pd
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Permitir requests desde el frontend

# Configuración de la base de datos
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///inversiones.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
db = SQLAlchemy(app)


# Modelo de la tabla de transacciones
class Transaccion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.Date, nullable=False)
    valor = db.Column(db.String(100), nullable=False)  # Nombre de la acción/valor
    tipo = db.Column(db.String(10), nullable=False)  # 'Compra' o 'Venta'
    precio_accion = db.Column(
        db.Float, nullable=False
    )  # Precio por acción en el momento
    cantidad_acciones = db.Column(
        db.Float, nullable=False
    )  # Cantidad de acciones (calculada)
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
            "precio_accion": self.precio_accion,
            "cantidad_acciones": self.cantidad_acciones,
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
        required_fields = ["fecha", "valor", "tipo", "precio_accion", "valor_operacion"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo requerido: {field}"}), 400

        # Calcular cantidad de acciones y valor final
        precio_accion = float(data["precio_accion"])
        valor_operacion = float(data["valor_operacion"])
        comisiones = float(data.get("comisiones", 0))

        # Calcular cantidad de acciones (valor_operacion / precio_por_accion)
        cantidad_acciones = valor_operacion / precio_accion

        if data["tipo"] == "Compra":
            valor_final = valor_operacion + comisiones
        else:  # Venta
            valor_final = valor_operacion - comisiones

        transaccion = Transaccion(
            fecha=datetime.strptime(data["fecha"], "%Y-%m-%d").date(),
            valor=data["valor"],
            tipo=data["tipo"],
            precio_accion=precio_accion,
            cantidad_acciones=cantidad_acciones,
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
                "cantidad_comprada": 0,
                "cantidad_vendida": 0,
                "precio_medio_compra": 0,
                "precio_medio_venta": 0,
                "posicion_actual": 0,
                "beneficio_realizado": 0,
                "num_operaciones": 0,
            }

        resumen = valores_resumen[valor]
        resumen["num_operaciones"] += 1

        if transaccion.tipo == "Compra":
            resumen["total_compras"] += transaccion.valor_final
            resumen["cantidad_comprada"] += transaccion.cantidad_acciones
            resumen["posicion_actual"] += transaccion.valor_final
        else:  # Venta
            resumen["total_ventas"] += transaccion.valor_final
            resumen["cantidad_vendida"] += transaccion.cantidad_acciones
            resumen["posicion_actual"] -= transaccion.valor_operacion
            resumen["beneficio_realizado"] += transaccion.valor_final

    # Calcular precios medios y beneficio realizado por valor
    for valor_data in valores_resumen.values():
        # Precio medio de compra
        if valor_data["cantidad_comprada"] > 0:
            valor_data["precio_medio_compra"] = (
                valor_data["total_compras"] / valor_data["cantidad_comprada"]
            )

        # Precio medio de venta
        if valor_data["cantidad_vendida"] > 0:
            valor_data["precio_medio_venta"] = (
                valor_data["total_ventas"] / valor_data["cantidad_vendida"]
            )

        # Beneficio realizado
        if valor_data["total_ventas"] > 0:
            # Calcular beneficio basado en precio medio de compra vs venta
            coste_acciones_vendidas = (
                valor_data["cantidad_vendida"] * valor_data["precio_medio_compra"]
            )
            valor_data["beneficio_realizado"] = (
                valor_data["total_ventas"] - coste_acciones_vendidas
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


@app.route("/api/import/excel", methods=["POST"])
def import_excel():
    """Importar transacciones desde archivo Excel"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No se encontró archivo"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No se seleccionó archivo"}), 400

        if not file.filename.lower().endswith((".xlsx", ".xls")):
            return (
                jsonify({"error": "Formato de archivo no válido. Use .xlsx o .xls"}),
                400,
            )

        # Leer Excel
        df = pd.read_excel(BytesIO(file.read()))

        # Validar columnas requeridas
        required_columns = [
            "Fecha",
            "Valor",
            "Tipo",
            "Precio_Accion",
            "Valor_Operacion",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return (
                jsonify(
                    {
                        "error": f'Columnas faltantes: {", ".join(missing_columns)}',
                        "columnas_requeridas": required_columns,
                        "columnas_encontradas": list(df.columns),
                    }
                ),
                400,
            )

        transacciones_importadas = 0
        errores = []

        for index, row in df.iterrows():
            try:
                # Validar y convertir datos
                fecha = pd.to_datetime(row["Fecha"]).date()
                valor = str(row["Valor"]).strip()
                tipo = str(row["Tipo"]).strip()
                precio_accion = float(row["Precio_Accion"])
                valor_operacion = float(row["Valor_Operacion"])
                comisiones = float(row.get("Comisiones", 0))

                # Validar tipo
                if tipo not in ["Compra", "Venta"]:
                    errores.append(
                        f'Fila {index + 2}: Tipo debe ser "Compra" o "Venta"'
                    )
                    continue

                # Calcular cantidad y valor final
                cantidad_acciones = valor_operacion / precio_accion
                valor_final = (
                    valor_operacion + comisiones
                    if tipo == "Compra"
                    else valor_operacion - comisiones
                )

                # Crear transacción
                transaccion = Transaccion(
                    fecha=fecha,
                    valor=valor,
                    tipo=tipo,
                    precio_accion=precio_accion,
                    cantidad_acciones=cantidad_acciones,
                    valor_operacion=valor_operacion,
                    comisiones=comisiones,
                    valor_final=valor_final,
                )

                db.session.add(transaccion)
                transacciones_importadas += 1

            except Exception as e:
                errores.append(f"Fila {index + 2}: {str(e)}")

        if transacciones_importadas > 0:
            db.session.commit()

        return (
            jsonify(
                {
                    "message": f"Importación completada",
                    "transacciones_importadas": transacciones_importadas,
                    "errores": errores,
                    "total_filas": len(df),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": f"Error al procesar archivo: {str(e)}"}), 400


@app.route("/api/analisis/precios/<valor>", methods=["GET"])
def get_analisis_precios(valor):
    """Obtener análisis de precios históricos para un valor específico"""
    transacciones = (
        Transaccion.query.filter_by(valor=valor).order_by(Transaccion.fecha).all()
    )

    if not transacciones:
        return jsonify({"error": f"No se encontraron transacciones para {valor}"}), 404

    # Análisis de precios
    precios_compra = [t.precio_accion for t in transacciones if t.tipo == "Compra"]
    precios_venta = [t.precio_accion for t in transacciones if t.tipo == "Venta"]

    # Evolución temporal de precios
    evolucion_precios = []
    for transaccion in transacciones:
        evolucion_precios.append(
            {
                "fecha": transaccion.fecha.isoformat(),
                "precio": transaccion.precio_accion,
                "tipo": transaccion.tipo,
                "cantidad": transaccion.cantidad_acciones,
            }
        )

    # Estadísticas
    estadisticas = {
        "valor": valor,
        "precio_compra_min": min(precios_compra) if precios_compra else 0,
        "precio_compra_max": max(precios_compra) if precios_compra else 0,
        "precio_compra_medio": (
            sum(precios_compra) / len(precios_compra) if precios_compra else 0
        ),
        "precio_venta_min": min(precios_venta) if precios_venta else 0,
        "precio_venta_max": max(precios_venta) if precios_venta else 0,
        "precio_venta_medio": (
            sum(precios_venta) / len(precios_venta) if precios_venta else 0
        ),
        "num_compras": len(precios_compra),
        "num_ventas": len(precios_venta),
        "evolucion_precios": evolucion_precios,
    }

    return jsonify(estadisticas)


@app.route("/api/template/excel", methods=["GET"])
def get_excel_template():
    """Descargar plantilla de Excel para importación"""
    from flask import send_file
    import tempfile

    # Crear DataFrame con estructura de ejemplo
    template_data = {
        "Fecha": ["2024-01-15", "2024-01-20", "2024-02-01"],
        "Valor": ["AAPL", "TSLA", "AAPL"],
        "Tipo": ["Compra", "Compra", "Venta"],
        "Precio_Accion": [150.25, 200.50, 155.75],
        "Valor_Operacion": [1502.50, 2005.00, 779.00],
        "Comisiones": [5.00, 7.50, 5.00],
    }

    df = pd.DataFrame(template_data)

    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        df.to_excel(tmp.name, index=False)
        return send_file(
            tmp.name,
            as_attachment=True,
            download_name="plantilla_inversiones.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
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

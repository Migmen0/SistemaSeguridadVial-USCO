"""Ejecuta la limpieza del CSV y guarda la tabla preparada y el informe de auditoría."""
from pathlib import Path
import json
from data_utils import cargar_csv, limpiar_datos

# La raíz del proyecto se obtiene desde la ubicación de este archivo.
RAIZ = Path(__file__).resolve().parents[1]
RUTA_ENTRADA = RAIZ / "data" / "raw" / "VEHICULOS_INVOLUCRADOS_EN_UN_ACCIDENTE_DE_TRANSITO_LEY_2251-2022_20261027.csv"
RUTA_SALIDA = RAIZ / "data" / "processed" / "vehiculos_limpios.csv"
RUTA_AUDITORIA = RAIZ / "reports" / "auditoria_limpieza.json"


def main() -> None:
    """Carga, limpia, guarda resultados e imprime un resumen verificable."""
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_AUDITORIA.parent.mkdir(parents=True, exist_ok=True)
    datos_originales = cargar_csv(RUTA_ENTRADA)
    datos_limpios, auditoria = limpiar_datos(datos_originales)
    datos_limpios.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")
    RUTA_AUDITORIA.write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LIMPIEZA FINALIZADA")
    print(f"CSV original: {RUTA_ENTRADA}")
    print(f"CSV limpio: {RUTA_SALIDA}")
    print(f"Auditoría: {RUTA_AUDITORIA}")
    print(json.dumps(auditoria, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

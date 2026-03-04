#El histórico realizado en el archivo HistoricoAnual.py debe ser actulizado cada mes para realizar pronósticos, el siguiente código tiene esto por objetivo.
import pandas as pd
import numpy as np

#Carpeta principal
Proyecto="/content/drive/MyDrive/Forecasting-V2"

#Seleccion de la sucursal
indice_sucursal=2
Rutas={
    "actualizacion": f"{Proyecto}/historico_2026",
    "sucursal": f"{Proyecto}/Suc{indice_sucursal}"
  }
#actulizacion: Lugar donde esta cargado el archivo con los datos del mes nuevo, por ejemplo: Enero
#sucursal: Carpeta de la sucursal en particular

#Subcarpetas
Subrutas={
    "datos": f"{Rutas["sucursal"]}/datos"
}
#Datos dentro de la sucursal 2, lugar donde están cargados los archivos obtenidos del historico completo/historico anterior

#Constantes
frecuencia="W-SUN"

#Actualización para datos de entrenamiento
ruta_Intermitente=f"{Subrutas["datos"]}/Intermitentes sucursal {indice_sucursal}.csv" #Datos de entrenamiento de Intermitentes
ruta_ML=f"{Subrutas["datos"]}/ML sucursal {indice_sucursal}.csv" #Datos de entrenamiento de ML
ruta_datos_nuevos=f"{Rutas["actualizacion"]}/Ventas Enero 2026.csv" #Historico del mes a agregar
ruta_promociones=f"{Rutas["actualizacion"]}/promociones enero 2026.csv" #Promociones dl mes a agregar

#Lectura de los 4 archivos
df_Intermitente=pd.read_csv(ruta_Intermitente) #Output para la sucursal n de Intermitentes en HistoricoAnual.py
df_ML=pd.read_csv(ruta_ML) #Output para la sucursal n de ML/Continuos en HistoricoAnual.py
df_nuevo=pd.read_csv(ruta_datos_nuevos,header=None) #Datos Nuevos de Historico
df_prom=pd.read_csv(ruta_promociones) #Promociones a agregar

#Primero se trabaja sobre el histórico nuevo
df_nuevo.columns=["sucursal","SKU","producto","ventas","fecha"]
df_nuevo=df_nuevo.groupby(["SKU","fecha","producto","sucursal"])["ventas"].sum().reset_index()
df_nuevo["fecha"]=pd.to_datetime(df_nuevo["fecha"],format="%Y-%m-%d")
df_nuevo["ventas"]=pd.to_numeric(df_nuevo["ventas"],errors="coerce")
df_nuevo["fecha_semana"]=df_nuevo["fecha"]+pd.to_timedelta((6-df_nuevo["fecha"].dt.day_of_week)%7,unit="D")

df_nuevo=(
    df_nuevo
    .groupby(["sucursal","SKU","fecha_semana"])
    .agg({"ventas":"sum","producto":"first"})
    .reset_index()
    .rename(columns={"fecha_semana":"fecha"})
    .sort_values(by=["fecha","SKU","sucursal"])
    .reset_index(drop=True)
)

df_nuevo=df_nuevo[["SKU","fecha","ventas","sucursal","producto"]].sort_values(by=["fecha","SKU"]).reset_index(drop=True)

#Promociones del histórico nuevo
df_prom.columns=["id_prom","prom","fecha_inicio","fecha_fin","sucursal","SKU","producto"]
df_prom["fecha_inicio"]=pd.to_datetime(df_prom["fecha_inicio"],format="%Y-%m-%d")
df_prom["fecha_fin"]=pd.to_datetime(df_prom["fecha_fin"],format="%Y-%m-%d")

df_prom["fecha"]=df_prom["fecha_inicio"]+pd.to_timedelta((6-df_prom["fecha_inicio"].dt.day_of_week)%7,unit="D")
df_prom["fin_calendario"]=df_prom["fecha_fin"]+pd.to_timedelta((6-df_prom["fecha_fin"].dt.day_of_week)%7,unit="D")

agg_prom={"prom":"first","id_prom":"first","fin_calendario":"last"}
df_prom=df_prom.groupby(["sucursal","SKU","fecha"]).agg(agg_prom).reset_index()

df_prom=df_prom[["id_prom","sucursal","SKU","fecha","fin_calendario","prom"]].sort_values(by=["id_prom","sucursal","SKU","fecha"]).reset_index(drop=True)

df_prom_especifica=df_prom[df_prom["sucursal"]!=0].reset_index(drop=True)
df_prom_general=df_prom[df_prom["sucursal"]==0].reset_index(drop=True)

df_nuevo_esp=pd.merge(
    df_nuevo,
    df_prom_especifica,
    on=["sucursal","SKU","fecha"],
    how="left"
)

df_nuevo_gen=pd.merge(
    df_nuevo,
    df_prom_general,
    on=["SKU","fecha"],
    how="left"
)

df_nuevo_final=df_nuevo_esp.copy()
df_nuevo_final["id_prom"]=df_nuevo_final["id_prom"].fillna(df_nuevo_gen["id_prom"])
df_nuevo_final["prom"]=df_nuevo_final["prom"].fillna(df_nuevo_gen["prom"])
df_nuevo_final["fin_calendario"]=df_nuevo_final["fin_calendario"].fillna(df_nuevo_gen["fin_calendario"])
df_nuevo_final=df_nuevo_final.sort_values(by=["fecha","sucursal","SKU"]).reset_index(drop=True)

!pip install utilsforecast

import utilsforecast as utils
from utilsforecast.preprocessing import fill_gaps
from datetime import datetime

#Se trabaja por sucursal para hacer los fill_gaps con utilsforecast
df_nuevo_sucursal=df_nuevo_final[df_nuevo_final["sucursal"]==indice_sucursal].reset_index(drop=True).drop(columns="sucursal")
coalescer=lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan")
coalescer_tiempo=lambda s: s.dropna().iloc[0] if s.notna().any() else pd.to_datetime("nan")

#Para intermitentes
df_nuevo_Intermitentes=df_nuevo_sucursal[df_nuevo_sucursal["SKU"].isin(df_Intermitente["SKU"].unique())]
df_nuevo_Intermitentes=fill_gaps(
    df_nuevo_Intermitentes,
    freq=frecuencia,
    start="global",
    id_col="SKU",
    time_col="fecha"
)

df_nuevo_Intermitentes["ventas"]=df_nuevo_Intermitentes["ventas"].fillna(0)
df_nuevo_Intermitentes["producto"]=df_nuevo_Intermitentes.groupby("SKU")["producto"].ffill()
df_nuevo_Intermitentes["producto"]=df_nuevo_Intermitentes.groupby("SKU")["producto"].bfill()
df_nuevo_Intermitentes=df_nuevo_Intermitentes.rename(columns={"fin_calendario":"fin_prom"})

df_Intermitente["fecha"]=pd.to_datetime(df_Intermitente["fecha"],format="%Y-%m-%d")
df_Intermitente["ventas"]=pd.to_numeric(df_Intermitente["ventas"],errors="coerce")
df_Intermitente["fin_prom"]=pd.to_datetime(df_Intermitente["fin_prom"],format="%Y-%m-%d",errors="coerce")

df_Int_concat=pd.concat([df_Intermitente,df_nuevo_Intermitentes],ignore_index=True)
dup_Int=df_Int_concat.duplicated(subset=["SKU","fecha"],keep=False)
df_Int_dup=df_Int_concat[dup_Int].copy().reset_index(drop=True)
df_Int_nodup=df_Int_concat[~dup_Int].copy().reset_index(drop=True)

df_Int_dup_res=(
    df_Int_dup
    .groupby(["SKU","fecha"])
    .agg({"ventas":"sum","producto":"first","id_prom":coalescer,"fin_prom":coalescer_tiempo,"prom":coalescer})
    .reset_index()
)

df_Intermitente_Actualizado=pd.concat([df_Int_nodup,df_Int_dup_res],ignore_index=True).sort_values(by=["SKU","fecha"]).reset_index(drop=True)

#df_Intermitente_Actualizado.to_csv("df_Intermitente_Actualizado.csv",index=False)
df_Intermitente_Actualizado.to_csv(f"{Subrutas["datos"]}/Intermitentes sucursal {indice_sucursal} actualizado.csv")
#Nota importante, el archivo debe ser nombrado como Intermitentes sucursal {indice_sucursal}.csv, SIN actualizado, de momento para que quede tanto el orignial
#como el actualizado se opto por poner ese nombre pero mes tras mes debe de ocupar el lugar que dejará el histórico sin el mes más reciente

df_nuevo_ML=df_nuevo_sucursal[df_nuevo_sucursal["SKU"].isin(df_ML["SKU"].unique())]

df_nuevo_ML=fill_gaps(
    df_nuevo_ML,
    freq=frecuencia,
    start="global",
    id_col="SKU",
    time_col="fecha"
)

df_nuevo_ML["ventas"]=df_nuevo_ML["ventas"].fillna(0)
df_nuevo_ML["producto"]=df_nuevo_ML.groupby("SKU")["producto"].ffill()
df_nuevo_ML["producto"]=df_nuevo_ML.groupby("SKU")["producto"].bfill()
df_nuevo_ML=df_nuevo_ML.rename(columns={"fin_calendario":"fin_prom"})

df_ML["fecha"]=pd.to_datetime(df_ML["fecha"],format="%Y-%m-%d")
df_ML["ventas"]=pd.to_numeric(df_ML["ventas"],errors="coerce")
df_ML["fin_prom"]=pd.to_datetime(df_ML["fin_prom"],format="%Y-%m-%d",errors="coerce")

df_ML_concat=pd.concat([df_ML,df_nuevo_ML],ignore_index=True)

dup_ML=df_ML_concat.duplicated(subset=["SKU","fecha"],keep=False)
df_ML_dup=df_ML_concat[dup_ML].copy().reset_index(drop=True)
df_ML_nodup=df_ML_concat[~dup_ML].copy().reset_index(drop=True)

df_ML_dup_res=(
    df_ML_dup
    .groupby(["SKU","fecha"])
    .agg({"ventas":"sum","producto":"first","id_prom":coalescer,"fin_prom":coalescer_tiempo,"prom":coalescer})
    .reset_index()
)

df_ML_Actualizado=pd.concat([df_ML_nodup,df_ML_dup_res],ignore_index=True).sort_values(by=["SKU","fecha"]).reset_index(drop=True)

df_ML_Actualizado.to_csv("df_ML_Enero.csv",index=False)
df_ML_Actualizado.to_csv(f"{Subrutas["datos"]}/ML sucursal {indice_sucursal} actualizado.csv")
#Mismo caso que la anterior nota el archivo debe ser nombrado como ML sucursal {indice_sucursal}.csv, SIN actualizado.

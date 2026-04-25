#En mi caso estoy realizando todo en Drive para poder mandar automaticamente la información sin necesidad de almacenarla localmente
import os
from google.colab import drive
drive.mount("/content/drive")

#Carpeta principal
Proyecto="/content/drive/MyDrive/Forecasting-V2"

#Seleccion de la sucursal
indice_sucursal=3
mes="Marzo"
Rutas={
    "sucursal": f"{Proyecto}/Suc{indice_sucursal}"
  }

#Subcarpetas
Subrutas={
    "modelos": f"{Rutas["sucursal"]}/modelos",
    "pronosticos": f"{Rutas["sucursal"]}/pronosticos",
    "metadatos": f"{Rutas["sucursal"]}/metadatos",
    "validacion": f"{Rutas["sucursal"]}/validacion",
    "datos": f"{Rutas["sucursal"]}/datos",
    "validacion_ventas_reales": f"{Rutas["sucursal"]}/validacion_ventas_reales"
}

for ruta in Subrutas.values():
    os.makedirs(ruta, exist_ok=True)

print("Carpetas listas en Drive")

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime

#Costantes relacionadas al forecasting:
frecuencia="W-SUN" #Importanticimo debe de coincidir con el formato de entrada
horizonte=4 #Pronóstico a 1 mes en el futuro

#Rutas de Entrada para Entrenamiento
ruta_Intermitente=f"{Subrutas["datos"]}/Intermitentes sucursal {indice_sucursal}.csv"
ruta_ML=f"{Subrutas["datos"]}/ML sucursal {indice_sucursal}.csv"

print(f"\nEsperando archivos en:")
print(f"{ruta_Intermitente}")
print(f"{ruta_ML}")

#Carga de los CSV
def cargar_csv(
    ruta_intermitente: str=ruta_Intermitente,
    ruta_ML: str=ruta_ML
) ->tuple[pd.DataFrame,pd.DataFrame]:
    def ord_datos(df: pd.DataFrame) -> pd.DataFrame:
      df=df.copy()
      df=df.rename(columns={
          "SKU":"unique_id",
          "ventas":"y",
          "fecha":"ds"
      })

      df["ds"]=pd.to_datetime(df["ds"],format="%Y-%m-%d")
      df["y"]=pd.to_numeric(df["y"],errors="coerce")
      df["unique_id"]=df["unique_id"].astype(str)
      df["id_prom"]=df["id_prom"].fillna(0).astype(int)
      df["fin_prom"]=pd.to_datetime(df["fin_prom"],format="%Y-%m-%d",errors="coerce")
      df["semanas_restantes_promo"]=(
          (df["fin_prom"]-df["ds"]).dt.days
          .clip(lower=0)
          .fillna(0)
          .div(7)
          .round(0)
          .astype(int)
      )
      return df.sort_values(by=["unique_id","ds"]).reset_index(drop=True)

    #Se agregan los features a futuro para los continuos
    def features_estacionalidad(df: pd.DataFrame)->pd.DataFrame:
        df=df.copy().sort_values(by=["unique_id","ds"]).reset_index(drop=True)
        semana_corr=df["ds"].dt.strftime('%U').astype(int)
        df["week"]=np.where(semana_corr==0,53,semana_corr)
        df["month"]=df["ds"].dt.month.astype(int)
        df["quarter"]=df["ds"].dt.quarter.astype(int)
        df["year"]=df["ds"].dt.year.astype(int)

        df["y_prom_semana"]=(
            df.groupby(["unique_id","week"])["y"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_semana"]=df["y_prom_semana"].fillna(
            df.groupby(["unique_id"])["y"].transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_mes"]=(
            df.groupby(["unique_id","month"])["y"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_mes"]=df["y_prom_mes"].fillna(
            df.groupby(["unique_id"])["y"].transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_trimestre"]=(
            df.groupby(["unique_id","quarter"])["y"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )

        df["y_prom_trimestre"]=df["y_prom_trimestre"].fillna(
            df.groupby(["unique_id"])["y"].transform(lambda x: x.shift(1).expanding().mean())
        )

        year_actual=df["ds"].dt.year.max()

        ventas_anuales_completas=(
            df[df["ds"].dt.year<year_actual]
            .groupby(["unique_id","year"])
            .agg({"y":"sum"})
            .reset_index()
            .rename(columns={"y":"ventas_anuales"})
        )

        ventas_anuales_completas["tendencia_anual"]=(
            ventas_anuales_completas.groupby("unique_id")["ventas_anuales"]
              .pct_change()
              .fillna(0)
          )

        return df

    df_intermitente=ord_datos(pd.read_csv(ruta_intermitente))
    df_ML_sin_ft=ord_datos(pd.read_csv(ruta_ML))
    df_ML=features_estacionalidad(df_ML_sin_ft)

    df_ML.to_csv(f"{Subrutas['datos']}/ML_con_ft.csv",index=False)
    print(f"CSV Intermitente cargado SKUs: {len(df_intermitente['unique_id'].unique()):,}")
    print(f"CSV ML cargado SKUs: {len(df_ML['unique_id'].unique()):,}")

    return df_intermitente,df_ML
    
!pip install mlforecast lightgbm

#Inicio de entrenamiento y pronóstico de Intermitentes
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from lightgbm import LGBMRegressor

def entrenar_adida(
    df_intermitente: pd.DataFrame,
    horizonte: int=horizonte,
) ->MLForecast:

    df_model=df_intermitente[["unique_id","ds","y"]].copy()
    df_model["unique_id"]=df_model["unique_id"].astype(int)

    #Features específicas para demanda intermitente
    # Proporción de semanas con ventas por SKU
    df_model["prop_ventas"]=(
        df_model.groupby("unique_id")["y"]
        .transform(lambda x:(x.shift(1)>0).expanding().mean())
    )

    #Semanas desde la última demanda positiva
    def semanas_desde_ultima_venta(serie):
        result=[]
        contador=0
        for val in serie.shift(1).fillna(0):
            if val>0:
                contador=0
            else:
                contador+=1
            result.append(contador)
        return pd.Series(result, index=serie.index)

    df_model["semanas_sin_venta"] = (
        df_model.groupby("unique_id")["y"]
        .transform(semanas_desde_ultima_venta)
    )

    lags=[4,8,12,52]

    lag_transforms={
        4: [RollingMean(window_size=4),RollingStd(window_size=4)],
        8: [RollingMean(window_size=8)],
        12: [RollingMean(window_size=12)]
     }

    mlf = MLForecast(
        models={
            "lgbm_intermitente": LGBMRegressor(
                objective="tweedie",   # diseñado para distribuciones con muchos ceros
                tweedie_variance_power=1.85,  # entre 1 (Poisson) y 2 (Gamma)
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        },
        freq=frecuencia,
        lags=lags,
        lag_transforms=lag_transforms
        )
    print("Ejecutando cross-validation")
    cv_df=mlf.cross_validation(
        df=df_model,
        h=horizonte,
        n_windows=4,
        step_size=horizonte,
        fitted=True,
        static_features=["unique_id"]
        )

    mae=(cv_df["lgbm_intermitente"]-cv_df["y"]).abs().mean()
    rmse=np.sqrt(((cv_df["lgbm_intermitente"]-cv_df["y"])**2).mean())

    metricas_df=pd.DataFrame([{
        "modelo":"LightGBM_Intermitente",
        "MAE":round(mae,6),
        "RMSE":round(rmse,6)
    }])

    cv_por_sku=(
        cv_df.groupby("unique_id")
        .apply(lambda g: pd.Series({
            "MAE": (g["lgbm_intermitente"]-g["y"]).abs().mean(),
            "RMSE": np.sqrt(((g["lgbm_intermitente"]-g["y"])**2).mean()),
            "bias": (g["lgbm_intermitente"]-g["y"]).mean()
        }),include_groups=False)
        .reset_index()
    )

    skus_riesgosos=cv_por_sku[cv_por_sku["MAE"]>mae*1.5]

    if len(skus_riesgosos)>0:
        print(f"Cantidad de SKUs con riesgo en el pronóstico de intermitentes: {len(skus_riesgosos)}, con un error 1.5 veces mayor al promedio ({mae:.2f})")
        print(skus_riesgosos.sort_values("MAE", ascending=False).to_string())
        skus_riesgosos.to_excel(f"{Subrutas['validacion']}/skus_riesgosos_Intermitentes_{mes}.xlsx",index=False)

    metricas_df.to_parquet(f"{Subrutas['validacion']}/cv_metricas_intermitente.parquet", index=False)
    cv_df.to_parquet(f"{Subrutas['validacion']}/cv_detalle_intermitente.parquet", index=False)
    cv_por_sku.to_parquet(f"{Subrutas['validacion']}/cv_metricas_por_sku_intermitente.parquet", index=False)

    mlf.fit(
        df_model,
        static_features=["unique_id"],
        fitted=True
        )
    joblib.dump(mlf,f"{Subrutas['modelos']}/mlf_intermitentes.joblib")
    df_model.to_parquet(f"{Subrutas['modelos']}/df_intermitentes.parquet",index=False)
    joblib.dump({
        "fecha_entrenamiento":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_SKUs":df_intermitente["unique_id"].nunique(),
        "horizonte":horizonte,
        "freq":frecuencia,
        "metricas_cv":metricas_df.to_dict(orient="records"),
        "skus":df_intermitente["unique_id"].unique().tolist()
     },f"{Subrutas['metadatos']}/metadatos_intermitentes.joblib")

    print("Modelo Intermitente ML guardado en Drive")
    return mlf

#Pronóstico de Intermitentes
def pronosticar_adida(
    df_intermitente: pd.DataFrame,
    horizonte: int=horizonte
) ->pd.DataFrame:

  mlf=joblib.load(f"{Subrutas['modelos']}/mlf_intermitentes.joblib")

  if df_intermitente is None:
    df_model=pd.read_parquet(f"{Subrutas['modelos']}/df_intermitentes.parquet")
  else:
    df_model=df_intermitente[["unique_id","ds","y"]].copy()

  df_futuro=mlf.make_future_dataframe(h=horizonte)
  df_futuro["unique_id"]=df_futuro["unique_id"].astype(int)

  ultima_prop=(
      df_intermitente.assign(
          prop_ventas=df_intermitente.groupby("unique_id")["y"].transform(lambda x: (x.shift(1)>0).expanding().mean())
      )
      .groupby("unique_id")["prop_ventas"].last().reset_index()
      )

  ultima_prop["unique_id"] = ultima_prop["unique_id"].astype(int)

  def calculo_semanas_sin_venta_actual(grupo):
      y=grupo.sort_values("ds")["y"].values
      contador=0
      for val in reversed(y):
          if val>0:
            break
            contador+=1
      return contador

  ultimas_semanas=(
      df_intermitente.groupby("unique_id")
      .apply(calculo_semanas_sin_venta_actual,include_groups=False)
      .reset_index()
      .rename(columns={0:"semanas_sin_venta"})
  )
  ultimas_semanas["unique_id"] = ultimas_semanas["unique_id"].astype(int)

  df_futuro=pd.merge(
      df_futuro,
      ultima_prop,
      on="unique_id",
      how="left"
  )

  df_futuro=pd.merge(
      df_futuro,
      ultimas_semanas,
      on="unique_id",
      how="left"
  )

  forecast_df=mlf.predict(h=horizonte,X_df=df_futuro)
  forecast_df=forecast_df.rename(columns={"lgbm_intermitente":"y_pred"})
  forecast_df["y_pred"]=forecast_df["y_pred"].clip(lower=0)

  forecast_df.to_parquet(f"{Subrutas['pronosticos']}/forecast_intermitente.parquet",index=False)

  return forecast_df

#Fin de entrenamiento y pronóstico de Intermitentes

#Inicio de entrenamiento y pronóstico de Continuos
def entrenar_mlforecast(
    df_ML: pd.DataFrame,
    horizonte: int=horizonte
) ->MLForecast:
    df_model=df_ML[["unique_id","ds","y","y_prom_semana","y_prom_mes","y_prom_trimestre"]].copy()
    df_model["unique_id"]=df_model["unique_id"].astype(int)

    lags=[4,8,12,52]

    lag_transforms={
        4: [RollingMean(window_size=4),RollingStd(window_size=4)],
        8: [RollingMean(window_size=8)],
        12: [RollingMean(window_size=12)],
        52: [RollingMean(window_size=52)]
    }

    mlf=MLForecast(
        models={
            "lgbm":LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        },
        freq=frecuencia,
        lags=lags,
        lag_transforms=lag_transforms
    )

    print("Ejecutando cross-validation")
    cv_df=mlf.cross_validation(
        df=df_model,
        h=horizonte,
        n_windows=4,
        step_size=horizonte,
        fitted=True,
        static_features=["unique_id"]
    )

    mae=(cv_df["lgbm"]-cv_df["y"]).abs().mean()
    rmse=np.sqrt(((cv_df["lgbm"]-cv_df["y"])**2).mean())

    metricas_df=pd.DataFrame([{
        "modelo":"LightGBM",
        "MAE":round(mae,6),
        "RMSE":round(rmse,6)
    }])

    cv_por_sku=(
        cv_df.groupby("unique_id")
        .apply(lambda g: pd.Series({
            "MAE": (g["lgbm"]-g["y"]).abs().mean(),
            "RMSE": np.sqrt(((g["lgbm"]-g["y"])**2).mean()),
            "bias": (g["lgbm"]-g["y"]).mean()
        }),include_groups=False)
        .reset_index()
    )

    #Skus con riesgo en el pronóstico con errores mayores al error promedio
    skus_riesgosos=cv_por_sku[cv_por_sku["MAE"]>mae*2]
    if len(skus_riesgosos)>0:
        print(f"Cantidad de SKUs con riesgo en el pronóstico: {len(skus_riesgosos)}, con un error 2 veces mayor al promedio ({mae:.2f})")
        print(skus_riesgosos.sort_values("MAE", ascending=False).to_string())
        skus_riesgosos.to_excel(f"{Subrutas['validacion']}/skus_riesgosos_ML_{mes}.xlsx",index=False)

    metricas_df.to_parquet(f"{Subrutas['validacion']}/cv_metricas_ML.parquet",index=False)
    cv_df.to_parquet(f"{Subrutas['validacion']}/cv_detalle_ML.parquet",index=False)
    cv_por_sku.to_parquet(f"{Subrutas['validacion']}/cv_metricas_por_sku_ML.parquet",index=False)

    mlf.fit(
        df_model,
        static_features=["unique_id"],
        fitted=True
    )

    joblib.dump(mlf,f"{Subrutas['modelos']}/mlf_ML.joblib")
    df_model.to_parquet(f"{Subrutas['modelos']}/df_ML.parquet",index=False)
    joblib.dump({
        "fecha_entrenamiento":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_SKUs":df_ML["unique_id"].nunique(),
        "horizonte":horizonte,
        "freq":frecuencia,
        "metricas_cv":metricas_df.to_dict(orient="records"),
        #"features_exogeneas": ["y_prom_semana","y_prom_mes","y_prom_trimestre","ultimo_crecimiento"],
        "features_exogeneas": ["y_prom_semana","y_prom_mes","y_prom_trimestre"],
        "skus":df_ML["unique_id"].unique().tolist()
    },f"{Subrutas['metadatos']}/metadatos_ML.joblib")

    print("Modelo ML guardado en Drive")

    importancia=mlf.models_["lgbm"].feature_importances_
    features=mlf.ts.features_order_
    print(pd.Series(importancia, index=features).sort_values(ascending=False))
    return mlf

#Pronostico de ML
def pronosticar_mlforecast(
    df_ML: pd.DataFrame,
    #df_futuro_exogeneas: pd.DataFrame,
    horizonte: int=horizonte
) ->pd.DataFrame:

    mlf=joblib.load(f"{Subrutas['modelos']}/mlf_ML.joblib")
    df_ML=df_ML[["unique_id","ds","y","week","month","quarter","y_prom_semana","y_prom_mes","y_prom_trimestre"]].copy()
    df_ML["unique_id"]=df_ML["unique_id"].astype(int)
    df_ML["ds"]=pd.to_datetime(df_ML["ds"])

    df_futuro=mlf.make_future_dataframe(h=horizonte)

    df_futuro["week"]=df_futuro["ds"].dt.strftime('%U').astype(int)
    df_futuro["month"]=df_futuro["ds"].dt.month.astype(int)
    df_futuro["quarter"]=df_futuro["ds"].dt.quarter.astype(int)

    prom_sem=df_ML.groupby(["unique_id","week"])["y"].mean().reset_index().rename(columns={"y":"y_prom_semana"})

    df_futuro=pd.merge(
        df_futuro,
        prom_sem,
        on=["unique_id","week"],
        how="left"
    )

    prom_mes=df_ML.groupby(["unique_id","month"])["y"].mean().reset_index().rename(columns={"y":"y_prom_mes"})

    df_futuro=pd.merge(
        df_futuro,
        prom_mes,
        on=["unique_id","month"],
        how="left"
    )

    prom_trim=df_ML.groupby(["unique_id","quarter"])["y"].mean().reset_index().rename(columns={"y":"y_prom_trimestre"})

    df_futuro=pd.merge(
        df_futuro,
        prom_trim,
        on=["unique_id","quarter"],
        how="left"
    )

    df_futuro=df_futuro.drop(columns=["week","month","quarter"],errors="ignore")

    forecast_df=mlf.predict(h=horizonte,X_df=df_futuro)
    forecast_df=forecast_df.rename(columns={"lgbm":"y_pred"})
    forecast_df["y_pred"]=forecast_df["y_pred"].clip(lower=0)

    forecast_df.to_parquet(f"{Subrutas['pronosticos']}/forecast_ML.parquet",index=False)
    return forecast_df

#Pipeline principal
def pipeline_completo(
    ruta_intermitente: str=ruta_Intermitente,
    ruta_ML: str=ruta_ML,
    horizonte: int=horizonte,
    calendario_promo: pd.DataFrame=None,
    solo_pronosticar: bool=False,
    entrenar_intermitente: bool=True,
    entrenar_continuo: bool=True,
    pron_adida: bool=True
) ->pd.DataFrame:
    inicio=datetime.now()
    print(f"Pipeline Iniciado: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Frecuencia: {frecuencia}")
    print(f"Horizonte: {horizonte} semanas")

    print("\n [1/4] Cargando archivos desde Drive")
    df_intermitente,df_ML=cargar_csv(ruta_intermitente,ruta_ML)

    if not solo_pronosticar:
        print("\n [2/4] Entrenando modelos")
        if entrenar_intermitente:
            print("\n StatsForecast Intermitentes")
            entrenar_adida(df_intermitente,horizonte)
        else:
            print("\n Entrenamiento de StatsForecast Omitido Usando modelo guardado")

        if entrenar_continuo:
            print("\n MLForecast")
            entrenar_mlforecast(df_ML,horizonte)
        else:
            print("\n Entrenamiento de MLForecast Omitido Usando modelo guardado")
    else:
        print("\n [2/4] Cargando modelos desde Drive, solo se harán pronósticos")

    print("\n [3/4] Generando pronósticos")
    if pron_adida:
      print("\n Generando pronósticos de Intermitentes")
      forecast_intermitente=pronosticar_adida(df_intermitente, horizonte)
    else:
      print("\n No se generarán pronósticos nuevos de intermitentes, usando resultados guardados")
      forecast_intermitente=pd.read_parquet(f"{Subrutas['pronosticos']}/forecast_intermitente.parquet")

    #futuro_exogeneas=const_futuro_exogeneas(df_ML,horizonte,calendario_promo)
    #forecast_ML=pronosticar_mlforecast(futuro_exogeneas, horizonte)
    print("\n Generando pronósticos de MLForecast")
    forecast_ML=pronosticar_mlforecast(df_ML,horizonte)

    print(f"\n [4/4] Generando resultados")
    nombre_map=pd.concat([
        df_intermitente.drop_duplicates("unique_id")[["unique_id","producto"]],
        df_ML.drop_duplicates("unique_id")[["unique_id","producto"]]
        ]).drop_duplicates("unique_id").reset_index(drop=True)

    forecast_intermitente["Modelo"]="Intermitente"
    forecast_ML["Modelo"]="MLForecast"

    forecast_ML["unique_id"]=forecast_ML["unique_id"].astype(str)
    forecast_intermitente["unique_id"]=forecast_intermitente["unique_id"].astype(str)

    forecast_total=(
        pd.concat([forecast_intermitente,forecast_ML],ignore_index=True)
        .merge(nombre_map,on="unique_id",how="left")
    )

    forecast_total.to_parquet(f"{Subrutas['pronosticos']}/forecast_total.parquet",index=False)
    forecast=forecast_total.copy()

    forecast["unique_id"]=pd.to_numeric(forecast["unique_id"],errors="coerce")

    reporte=(
        forecast
        .groupby(["unique_id"])
        .agg({"y_pred":"sum","Modelo":"first","producto":"first"})
        .reset_index()
        .rename(columns={"y_pred":"Pronóstico de Ventas","unique_id":"SKU","producto":"Producto"})
        .sort_values(by="SKU")
        .reset_index(drop=True)
    )

    reporte["Pronóstico de Ventas"]=np.round(reporte["Pronóstico de Ventas"])
    reporte.to_parquet(f"{Subrutas['pronosticos']}/reporte_total.parquet",index=False)

    reporte_excel=reporte.drop(columns=["Modelo"]).copy()
    reporte_excel.to_excel(f"{Subrutas['pronosticos']}/Reporte de pronósticos {mes}.xlsx",index=False)

    duracion=(datetime.now()-inicio).seconds
    print(f"Pipeline Compleado Duración: {duracion} s")
    print(f"Archivo guardado en: {Subrutas['pronosticos']}/reporte_total.parquet")
    return forecast_total

#Aquí es donde selecciono el tipo de pronóstico
#Entrenamiento y pronóstico
resultado=pipeline_completo()

#Solo repronóstico
#resultado=pipeline_completo(solo_pronosticar=True)

#Entrenamiento de continuos sin intermitentes
#resultado=pipeline_completo(entrenar_intermitente=False,entrenar_continuo=True)

#Entrenamiento de intermitentes sin continuos
#resultado=pipeline_completo(entrenar_adida=True,entrenar_mlforecast=False)

#Entrenamiento de continuos y pronóstico de continuos
#resultado=pipeline_completo(entrenar_intermitente=False,entrenar_continuo=True,pron_adida=False)

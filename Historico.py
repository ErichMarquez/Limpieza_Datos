import pandas as pd
import numpy as np

#Definir sobre que sucursal se trabajará
indice_sucursal=2

#Agregación del histórico
hist_23_25=pd.read_csv('Ventas_Historico.csv',header=None) #Aquí va el histórico desde 2023 hasta el domingo deseado de 2026
hist_23_25.columns=['SKU','ventas','sucursal','fecha']
hist_23_25=hist_23_25[hist_23_25['sucursal']==indice_sucursal].reset_index(drop=True)

nombres=pd.read_csv('CatalogoProductos.csv') #Aquí va el inventario con el cual se cruzan los nombres
nombres.columns=['Sucursal','SKU','producto','inventario']
nombres=nombres[nombres['Sucursal']==indice_sucursal].drop(columns=['Sucursal','inventario']).reset_index(drop=True)
hist_23_25=pd.merge(hist_23_25,nombres,on=['SKU'],how='left')
#Como resultado se obtiene el dataframe con nombres incluidos, los formatos de los archivos son tal cual los obtenemos de las vistas.

hist_23_25_sum=hist_23_25.groupby(['SKU','fecha','producto','sucursal'])['ventas'].sum().reset_index()
hist_23_25_sum['fecha']=pd.to_datetime(hist_23_25_sum['fecha'],format="%Y-%m-%d")
hist_23_25_sum['ventas']=pd.to_numeric(hist_23_25_sum['ventas'],errors='coerce')
hist_23_25_sum['fecha_semanal']=hist_23_25_sum['fecha']+pd.to_timedelta((6-hist_23_25_sum['fecha'].dt.day_of_week)%7,unit='D')

hist_23_25_semanal=(
    hist_23_25_sum
    .groupby(['sucursal','SKU','fecha_semanal'])
    .agg({'ventas':'sum','producto':'first'})
    .reset_index()
    .rename(columns={'fecha_semanal':'fecha'})
    .sort_values(by=['fecha','SKU'])
    .reset_index(drop=True)
)

hist_23_25_comp=hist_23_25_semanal[['SKU','fecha','ventas','sucursal','producto']].sort_values(by=['fecha','SKU']).reset_index(drop=True)

#Eliminación de SKUs Temporales
exclusiones=pd.read_csv('Exclusiones.txt')
exclusiones.columns=['SKU_Excluidos']
hist_23_25_comp['Exclusiones']=hist_23_25_comp['SKU'].isin(exclusiones['SKU_Excluidos'])

hist_excl=hist_23_25_comp[hist_23_25_comp['Exclusiones']==False].drop(columns=['Exclusiones']).reset_index(drop=True)
hist_excl['ventas']=hist_excl['ventas'].clip(lower=0)

#Agregación de Promociones
prom=pd.read_csv('Promociones.csv')
prom.columns=['id_prom','prom','fecha_inicio','fecha_fin','sucursal','SKU','producto']
prom['fecha_inicio']=pd.to_datetime(prom['fecha_inicio'], format="%Y-%m-%d")
prom['fecha_fin']=pd.to_datetime(prom['fecha_fin'], format="%Y-%m-%d")

prom['fecha']=prom['fecha_inicio']+pd.to_timedelta((6-prom['fecha_inicio'].dt.day_of_week)%7,unit='D')
prom['fin_calendario']=prom['fecha_fin']+pd.to_timedelta((6-prom['fecha_fin'].dt.day_of_week)%7,unit='D')

agg_prom={'prom':'first','id_prom':'first','fin_calendario':'last'}
prom_sindup=(
    prom.
    groupby(['sucursal','SKU','fecha']).
    agg(agg_prom).
    reset_index()
)

prom_sindup=prom_sindup[['id_prom','sucursal','SKU','fecha','fin_calendario','prom']].sort_values(by=['id_prom','sucursal','SKU','fecha']).reset_index(drop=True)

prom_especifica=prom_sindup[prom_sindup['sucursal']!=0].reset_index(drop=True)
prom_general=prom_sindup[prom_sindup['sucursal']==0].drop(columns=['sucursal']).reset_index(drop=True)
hist_prom_esp=pd.merge(
    hist_excl,
    prom_especifica,
    on=['SKU','fecha','sucursal'],
    how='left'
)

hist_prom_gen=pd.merge(
    hist_excl,
    prom_general,
    on=['SKU','fecha'],
    how='left'
)

Hist_Final=hist_prom_esp.copy()
Hist_Final['id_prom']=Hist_Final['id_prom'].fillna(hist_prom_gen['id_prom'])
Hist_Final['prom']=Hist_Final['prom'].fillna(hist_prom_gen['prom'])
Hist_Final['fin_calendario']=Hist_Final['fin_calendario'].fillna(hist_prom_gen['fin_calendario'])
Hist_Final=Hist_Final.sort_values(by=['fecha','sucursal','SKU']).reset_index(drop=True)

#Agregado semanal en formato W_SUN
!pip install utilsforecast
import utilsforecast as utils
from utilsforecast.preprocessing import fill_gaps
from datetime import datetime

Historico_Sucursal=Hist_Final[Hist_Final['sucursal']==indice_sucursal].drop(columns='sucursal').reset_index(drop=True)

Historico_Sucursal['fecha']=pd.to_datetime(Historico_Sucursal['fecha'])

His_Sucursal_Semanales=fill_gaps(
    df=Historico_Sucursal,
    freq='W-SUN',
    id_col='SKU',
    time_col='fecha'
)

His_Sucursal_Semanales['ventas']=His_Sucursal_Semanales['ventas'].fillna(0)
His_Sucursal_Semanales['producto']=His_Sucursal_Semanales.groupby('SKU')['producto'].ffill()

#Se agrega nuevo preprocessing: Filtración de productos descontinuados
semanas_sin_venta=16 #Hasta 4 meses sin que se resurta o venda un item

Historico_Sucursal=His_Sucursal_Semanales.copy()
Historico_Sucursal['fecha']=pd.to_datetime(Historico_Sucursal['fecha'])
fecha_corte=Historico_Sucursal["fecha"].max()
ventana=pd.Timedelta(weeks=semanas_sin_venta)
fecha_inicio_ventana=fecha_corte-ventana

ventas_recientes=(
    Historico_Sucursal[Historico_Sucursal["fecha"] >= fecha_inicio_ventana]
    .groupby("SKU")
    .agg({"ventas": "sum"})
    .reset_index()
    .rename(columns={"ventas": "ventas_recientes"})
).fillna(0)

todos_SKUs=Historico_Sucursal[["SKU"]].drop_duplicates().reset_index(drop=True)

ventas_recientes=pd.merge(
    todos_SKUs,
    ventas_recientes,
    on="SKU",
    how="left"
)

mask_descontinuados=(ventas_recientes["ventas_recientes"]==0)
skus_activos=ventas_recientes.loc[~mask_descontinuados,"SKU"].tolist()
skus_descontinuados=ventas_recientes.loc[mask_descontinuados,"SKU"].tolist()

df_activo=Historico_Sucursal[Historico_Sucursal["SKU"].isin(skus_activos)].copy().reset_index(drop=True)
df_descontinuado=Historico_Sucursal[Historico_Sucursal["SKU"].isin(skus_descontinuados)].copy().reset_index(drop=True)

df_descontinuado.to_parquet(f"{Subrutas["metadatos"]}/skus_descontinuados.parquet",index=False)

His_Sucursal_Semanales=df_activo.copy()

#Nuevas selecciones de categoria
LI_Sem=104
LS_Sem=110
Porc_ML=0.60

His_Sucursal_Semanales['Venta Binaria']=(His_Sucursal_Semanales['ventas']>0).astype(int)

Contador_Ventas=(
    His_Sucursal_Semanales
    .groupby('SKU')
    .agg(
        Producto=('producto','first'),
        Semanas_Con_Venta=('Venta Binaria','sum'),
        Semanas_Totales=('fecha','count')
    )
    .reset_index(
    )
)

Contador_Ventas['Porcentaje_Ventas']=Contador_Ventas['Semanas_Con_Venta']/Contador_Ventas['Semanas_Totales']
Contador_Ventas['Modelo Seleccionado']=np.select(
    [
        (Contador_Ventas['Semanas_Con_Venta']>=LS_Sem) & (Contador_Ventas['Porcentaje_Ventas']>=Porc_ML),
        (Contador_Ventas['Semanas_Totales']>=LI_Sem) & (Contador_Ventas['Semanas_Con_Venta']<LS_Sem)
    ],
    ['MLForecast','Método Intermitente'],
    default='Sin pronóstico'
)

SKU_NoViables=(Contador_Ventas[Contador_Ventas['Modelo Seleccionado']=='Sin pronóstico'])
SKU_Intermitentes=Contador_Ventas[Contador_Ventas['Modelo Seleccionado']=='Método Intermitente']
SKU_ML=Contador_Ventas[Contador_Ventas['Modelo Seleccionado']=='MLForecast']

#Extracción de Datos de Entrenamiento
df_ML=His_Sucursal_Semanales[His_Sucursal_Semanales['SKU'].isin(SKU_ML['SKU'])].drop(columns='Venta Binaria').rename(columns={'fin_calendario':'fin_prom'}).copy().reset_index(drop=True)

df_ML.to_csv('ML sucursal '+str(indice_sucursal)+'.csv',index=False)

df_Intermitentes=His_Sucursal_Semanales[His_Sucursal_Semanales['SKU'].isin(SKU_Intermitentes['SKU'])].drop(columns='Venta Binaria').rename(columns={'fin_calendario':'fin_prom'}).copy().reset_index(drop=True)

df_Intermitentes.to_csv('Intermitentes sucursal '+str(indice_sucursal)+'.csv',index=False)

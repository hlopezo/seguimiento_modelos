#funcion que nos devuleve nuestra matriz o cuadro comparativo de indicadores del modelo y variables
def cuadro_resumen(data): # cambia por base
    #####################################################################################
    #                  Creamos la Funcion divisora de la base de datos                  #
    #####################################################################################
    def df_split(df):
        df_split = []
        productos_interesantes = ["Taxi Pacifico", "Remisse", "Taxi", "Taxi Banca Consejero", "Recolocado", "Recolocado LD", "Consumo", "Consumo Alfin"]
        if df['producto'].isin(productos_interesantes).any():
            df_split.append(df[(df['base'] == 'entrenamiento') & (df['insuficiente'] == 0)])  # train 1 (solo consumo, recolocado, taxi)
            df_split.append(df[(df['base'] == 'validacion_interna') & (df['insuficiente'] == 0)])  # test 2 (solo consumo, recolocado, taxi)
            df_split.append(df[(df['base'] == 'entrenamiento') & (df['insuficiente'] == 0) | (df['base'] == 'validacion_interna') & (df['insuficiente'] == 0)])  # Total 4 (solo consumo, recolocado, taxi)
            df_split.append(df[(df['base'] == 'seguimiento') & (df['insuficiente'] == 0)])  # seguimiento 3  (solo consumo, recolocado, taxi)
        else:
            df_split.append(df[(df['base'] == 'entrenamiento') & (df['indeterminado'] == 0) & (df['insuficiente'] == 0)])  # train 1
            df_split.append(df[(df['base'] == 'validacion_interna') & (df['indeterminado'] == 0) & (df['insuficiente'] == 0)])  # test 2
            df_split.append(df[(df['base'] == 'entrenamiento') & (df['indeterminado'] == 0) & (df['insuficiente'] == 0) | (df['base'] == 'validacion_interna') & (df['indeterminado'] == 0) & (df['insuficiente'] == 0)])  # Toda la base
            df_split.append(df[(df['base'] == 'seguimiento') & (df['indeterminado'] == 0) & (df['insuficiente'] == 0)])  # seguimiento 3 
        return df_split

    df1,df2,df4,df3=df_split(data)
    # para las metricas como AOC y gini
    #####################################################################################
    #                  Creamos la Funcion Curva ROC                                     #
    #####################################################################################
    from sklearn.metrics import roc_auc_score, roc_curve
       # hasta aca es la funcion
    def plot_roc_curve(y, y_proba, label = ''):
        auc_roc = roc_auc_score(y, y_proba)
        fpr, tpr, thresholds = roc_curve(y, y_proba)
    #####################################################################################
    #                  Calculo de ROC Y Gini                                            #
    #####################################################################################
    Roc=[] # Creamos una lista vacia para llenarlo en el For
    Gini=[] # Creamos una lista vacia para llenarlo en el For
    for i in [df1, df2, df3]:
        unique_classes = i.tm.unique()
        if len(unique_classes) < 2:
            Roc.append(0)
            Gini.append(0) 
        else:
            plot_roc_curve(i.tm, i.pd, label=str(i['base'].unique()))
            Roc.append(round(roc_auc_score(i.tm, i.pd), 4))
            Gini.append(round((round(roc_auc_score(i.tm, i.pd), 4) * 2 - 1), 4)) 
    #####################################################################################
    #                  Creamos la Funcion KS                                           #
    #####################################################################################
    # funcion para indicador de ks
    def indicador_ks(datos):
        GI = datos[datos.tm == 0]
        BI = datos[datos.tm == 1]
        # Obtener porcentiles
        percentiles = np.percentile(datos.pd, np.arange(0, 101, 10))
        # Calcular frecuencias acumulativas para Ks_goods y Ks_bads
        ks_goods = np.histogram(GI.pd, bins=percentiles)[0]
        ks_bads = np.histogram(BI.pd, bins=percentiles)[0]
        # Calcular Ks_all y totales
        ks_all = ks_goods + ks_bads
        total_goods, total_bads = sum(ks_goods), sum(ks_bads)
        # Calcular porcentajes acumulativos
        ks_per_goods = np.cumsum(ks_goods) / total_goods if total_goods != 0 else np.zeros(10)
        ks_per_bads = np.cumsum(ks_bads) / total_bads if total_bads != 0 else np.zeros(10)
        # Calcular Ks
        ks = np.abs(ks_per_goods - ks_per_bads) if total_goods != 0 and total_bads != 0 else np.zeros(10)
        return round(max(ks), 4)
    ## hasta aca es la funcion
    # creamos una funcion q asigne la funcion a cada variaable podemos usar un apply tambien 
    def asignador(funcion_aplicadora):
        x=[]
        datos=[df1,df2,df3]
        for i in datos:
            x.append(funcion_aplicadora(i))
        return x
    ks=asignador(indicador_ks)
    ## hasta aca es la funcion
    # Creamos una funcion que nos diga el t discriminante
    #####################################################################################
    #                  Creamos la Funcion Test Discriminante                            #
    #####################################################################################
    def Tdis(datos):
        X_good = np.mean(datos.pd[datos.tm == 0])
        std_good = np.std(datos.pd[datos.tm == 0])
        X_bad = np.mean(datos.pd[datos.tm == 1])
        std_bad = np.std(datos.pd[datos.tm == 1])
        numerator = np.abs((X_good - X_bad) ** 2)
        denominator = std_good ** 2 - std_bad ** 2
        t_discriminante = np.sqrt(numerator / denominator * 2) if denominator != 0 else 0
        return round(t_discriminante, 4)
    TD=asignador(Tdis)
    # creamos la segmentacion por deciles
    
    def segmen(datos, datos1):
        percentiles = np.arange(0, 101, 10)
        def calcular_segmento(data):
            return np.histogram(data, bins=np.percentile(datos1.pd, percentiles))[0]
        psitrack = calcular_segmento(datos.pd)
        psimodel = calcular_segmento(datos1.pd)
        per_model = psimodel / np.sum(psimodel)
        per_track = psitrack / np.sum(psitrack)

        return [per_model.tolist(), per_track.tolist()]

    def PSI(A, B):
        A, B = segmen(A, B)
        PSI = [np.log(abs(A[i]) / abs(B[i])) * (A[i] - B[i]) if B[i] != 0 else 0 for i in range(10)]
        return round(sum(PSI), 4)
    def esta_ks(A, B):
        A, B = segmen(A, B)
        KS = [abs(A[i] - B[i]) for i in range(10)]
        return round(max(KS), 4)
    def herfindal(A, B):
        A, B = segmen(A, B)
        H_A = [pow(A[i], 2) for i in range(10)]
        H_B = [pow(B[j], 2) for j in range(10)]
        return round(sum(H_B), 4)
    ######################################################################################
    #                 Coeficiente de asimetria y Kurtosis                                #
    ######################################################################################    
    psi=[0,PSI(df2,df1),PSI(df3,df2)]
    esta_ks=[0,esta_ks(df2,df1),esta_ks(df3,df2)]
    IH=[herfindal(df1,df1),herfindal(df2,df1),herfindal(df3,df2)]
    M3=[] #Asimetria
    M4=[] #Kurtosis
    datos=[df1,df2,df3]
    for i in datos:
        M3.append(round(i.pd.skew(),4))
        M4.append(round(i.pd.kurt()+3,4))  
    ######################################################################################
    #                 tablas de indicadores de train, test y seguimiento                 #
    ######################################################################################
    name=['Entrenamiento','Validacion Interna','Seguimiento']
    df = pd.DataFrame(list(zip(name,Gini,Roc,ks,TD,psi,esta_ks,IH,M3,M4)), 
                      columns = ['Base','Indicador Gini','Indicador ROC','indicador KS','Test Discriminante',"Índice de Estabilidad Poblacional (PSI)",'KS',"Índice de Herfindahl (IH)","Coeficiente de Asimetría", "Coeficiente de Curtosis"])    
    
    minimo=[min(df1.iloc[:, 0]),min(df2.iloc[:, 0]),min(df3.iloc[:, 0])]
    maximo=[max(df1.iloc[:, 0]),max(df2.iloc[:, 0]),max(df3.iloc[:, 0])]
    cant=[len(df1.iloc[:, 0]),len(df2.iloc[:, 0]),len(df3.iloc[:, 0])]
    tm=[round(df1['tm'].mean()*100,2),round(df2['tm'].mean()*100,2),round(df3['tm'].mean()*100,2)]
    prde=[round(df1['pd'].mean()*100,2),round(df2['pd'].mean()*100,2),round(df3['pd'].mean()*100,2)]
    name=['Entrenamiento','Validacion Interna','Seguimiento']
    name=['Entrenamiento','Validacion Interna','Seguimiento']
    df_v = pd.DataFrame(list(zip(name,minimo,maximo,cant,tm,prde)), 
                        columns = ['Base','Minimo','Maximo','Numero de Creditos','tasa de malos promedio','PD promedio'])
    df_v.style.hide_index()
    df_unida=pd.concat([df4,df3],axis=0)
    df_c= pd.DataFrame(df_unida.iloc[:,0].unique(),columns = ['cosecha'])
    cantidad=[]
    tasa_malo=[]
    ProbDef=[]
    for i in df_c.cosecha:
        cantidad.append(len(df_unida.iloc[:,0][df_unida.iloc[:,0]==i]))
        tasa_malo.append(round((df_unida['tm'][df_unida.iloc[:,0]==i].mean()),4))
        ProbDef.append(round((df_unida['pd'][df_unida.iloc[:,0]==i].mean()),4))
    df_c['cantidad']=cantidad
    df_c['tasa malo']=tasa_malo
    df_c['pd']=ProbDef
    df_c['cosecha']=df_c['cosecha'].astype(str)


    return [df, df_v,df_c]

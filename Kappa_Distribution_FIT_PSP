from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


# Substitua 'seuarquivo.cdf' pelo caminho e nome do seu arquivo CDF
arquivo_cdf = '/home/benetti/Documentos/Instabilidade modulacional/Python/psp_swp_spe_sf0_l3_pad_20230321_v04'
arquivo_cdf2 = '/home/benetti/Documentos/Instabilidade modulacional/Python/psp_swp_spc_l3i_20230321_v02'
# Nome da variável que você deseja acessar
nome_variavel_energia = 'ENERGY_VALS'

# Abra o arquivo CDF
cdf = pycdf.CDF(arquivo_cdf)
cdf2 = pycdf.CDF(arquivo_cdf2)
# Verifique se a variável existe no arquivo
if nome_variavel_energia in cdf:
    # Obtenha os valores de energia em elétron-volts
    valores_energia_ev = cdf[nome_variavel_energia][0]

    # Converta os valores de energia para joules
    massa_eletron_kg = 9.10938356e-31
    valores_energia_joules = valores_energia_ev * 1.602176634e-19  # 1 eV = 1.602176634e-19 Joules

    # Calcule as velocidades em metros por segundo
    velocidades_ms = np.sqrt(2 * valores_energia_joules / massa_eletron_kg)* 1e-6

    # Salve os valores de energia e velocidade em um arquivo TXT
    arquivo_energia_velocidade = '/home/benetti/Documentos/Instabilidade modulacional/Python/energia_velocidade.txt'
    np.savetxt(arquivo_energia_velocidade, np.column_stack((valores_energia_ev, velocidades_ms)),
               delimiter='\t')

    print(f'Valores de energia e velocidade salvos em {arquivo_energia_velocidade}')
else:
    print(f'A variável {nome_variavel_energia} não foi encontrada no arquivo.')



# Nome da variável que você deseja acessar
nome_variavel = 'EFLUX_VS_PA_E'
nome_variavel2 = 'wp_fit'
#nome_variavel2 = 'T_TENSOR_INST'
#nome_variavel2 = 'TEMP'
# Ângulo fixo
angulo_fixo = 1


# Verifique se a variável existe no arquivo
if nome_variavel in cdf:
    # Obtenha os tempos correspondentes
   
    tempos_sub = cdf['Epoch'][:]
        
    # Acesse a variável
    variavel = cdf[nome_variavel]

    # Obtenha os dados da variável para o ângulo fixo
    dadossub = variavel[:, angulo_fixo - 1, :]
    
    mascara_subamostragem = np.arange(len(tempos_sub)) % 0 == 0

# Aplicar a máscara para obter os tempos e dados desejados
    tempos = tempos_sub[mascara_subamostragem]
    dados = dadossub[mascara_subamostragem]


    # Transponha os dados para que as energias se tornem colunas
    dados = dados.T

    # Adicione os valores de tempo formatados como a primeira linha em cada coluna
    tempos_formatados3 = [f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" for dt in tempos]
    
    tempos_formatados = tempos_formatados3[1:]
    dados_com_tempo = np.vstack((dados))
    
    # Salve os dados em um arquivo TXT
    arquivo_txt = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_exportados_angulo.txt'
    np.savetxt(arquivo_txt, dados_com_tempo, delimiter='\t', fmt='%s')

    print(f"Dados exportados para {arquivo_txt}")
else:
    print(f"A variável {nome_variavel} não foi encontrada no arquivo.")

nome_variavel2 = 'wp_fit'
if nome_variavel2 in cdf2:
    # Obtenha os tempos correspondentes
    tempos2 = cdf2['Epoch'][:]

    # Acesse a variável
    #variavel2 = cdf2[nome_variavel2][...]
    variavel2 = cdf2[nome_variavel2]
    #dados3 = variavel2[:,1]
    # Obtenha os dados da variável para o ângulo fixo
    dados3 = np.array(variavel2)
    dados2 = ((((dados3-6)*1000)**2)*1.67e-27)/(2*1.38e-23)
    #dados2 = dados3*11604
    # Adicione os valores de tempo formatados como a primeira linha em cada coluna
    tempos_formatados2 = [f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" for dt in tempos2]

    # Modificação aqui: Crie o DataFrame com duas linhas
    df_ti = pd.DataFrame(data=[tempos_formatados2,dados2], columns=tempos_formatados2)
    
    
nome_variavel3 = 'wp_fit_uncertainty'
if nome_variavel3 in cdf2:
   
    variavel3 = cdf2[nome_variavel3]
    dados5 = np.array(variavel3)
    dados4 = ((((dados5)*1000)**2)*1.67e-27)/(2*1.38e-23)
   
    df_ti5 = pd.DataFrame(data=[tempos_formatados2,dados4], columns=tempos_formatados2)

    # Salvar o DataFrame em um único arquivo TXT
    #arquivo_final2 = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_combinadosti.txt'
    #df_ti.to_csv(arquivo_final2, sep='\t', index=False)

    #print(f'Dados combinados salvos em {arquivo_final2}')

    # Salve os dados em um arquivo TXT
    #arquivo2_txt = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_t.txt'
    #np.savetxt(arquivo2_txt, df_ti.values, delimiter='\t', fmt='%s')

    #print(f"Dados exportados para {arquivo2_txt}")
#else:
    #print(f"A variável {nome_variavel2} não foi encontrada no arquivo.")
    
    
def convert_str_to_datetime(tempo_str):
    return datetime.strptime(tempo_str, '%H:%M:%S')
tempos_formatados_dt = [convert_str_to_datetime(tempo) for tempo in tempos_formatados]
tempos_formatados2_dt = [convert_str_to_datetime(tempo) for tempo in tempos_formatados2]
temperaturas_associadas = []
temperaturas_associadas2 = []

for tempo_dt in tempos_formatados_dt:
    # Encontrar o índice do valor mais próximo em tempos_formatados2
    indices_mais_proximos = np.argsort(np.abs(np.array(tempos_formatados2_dt) - tempo_dt))
    
    # Pegar o índice do valor mais próximo
    indice_mais_proximo = indices_mais_proximos[0]
   
    # Verificar se os valores são iguais
    if tempos_formatados2_dt[indice_mais_proximo] == tempo_dt:
        temperatura_associada = df_ti.iloc[1, indice_mais_proximo-1]
        temperatura_associada2 = df_ti5.iloc[1, indice_mais_proximo-1]
    else:
        # Calcular a média dos valores de temperatura entre os valores anteriores e posteriores
        indice_anterior = indices_mais_proximos[1] if len(indices_mais_proximos) > 1 else indice_mais_proximo
        indice_posterior = indices_mais_proximos[2] if len(indices_mais_proximos) > 2 else indice_mais_proximo
        temperatura_anterior = df_ti.iloc[1, indice_anterior]
        temperatura_posterior = df_ti.iloc[1, indice_posterior-1]
        temperatura_associada = (temperatura_anterior + temperatura_posterior) / 2
        temperatura_anterior2 = df_ti5.iloc[1, indice_anterior]
        temperatura_posterior2 = df_ti5.iloc[1, indice_posterior-1]
        temperatura_associada2 = (temperatura_anterior2 + temperatura_posterior2) / 2
    
    temperaturas_associadas.append(temperatura_associada)
    temperaturas_associadas2.append(temperatura_associada2)
    #temperaturas_associadas2 = np.array(temperaturas_associadas2)
df_ti2 = pd.DataFrame(data=[temperaturas_associadas], columns=tempos_formatados)

    # Salvar o DataFrame em um único arquivo TXT
arquivo_final3 = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_combinadosti2.txt'
df_ti2.to_csv(arquivo_final3, sep='\t', index=False)

print(f'Dados combinados salvos em {arquivo_final3}')

    


# Função para calcular as densidades no espaço de fase
def calcular_densidades(valores_fluxo, valores_energia_joules, massa_eletron_kg):
    return (valores_fluxo * 10000 * (massa_eletron_kg ** 2)) / (2 * (valores_energia_joules**2))


# Nome da variável que você deseja acessar
nome_variavel_energia = 'ENERGY_VALS'
nome_arquivo_fluxo_energia = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_exportados_angulo.txt'

# Verifique se a variável de energia existe no arquivo
if nome_variavel_energia in cdf:
    # Obtenha os valores de energia em elétron-volts
    valores_energia_ev = cdf[nome_variavel_energia][0]

    # Converta os valores de energia para joules
    valores_energia_joules = valores_energia_ev * 1.602176634e-19  # 1 eV = 1.602176634e-19 Joules

    # Carregue os dados do arquivo de fluxo de energia
    dados_fluxo_energia = np.loadtxt(nome_arquivo_fluxo_energia, delimiter='\t')

    # Calcule as densidades no espaço de fase para cada coluna
    massa_eletron_kg = 9.10938356e-31
    densidades_espaco_fase = np.apply_along_axis(calcular_densidades, 0, dados_fluxo_energia[:, 1:], valores_energia_joules, massa_eletron_kg)

    dados_para_salvar = np.column_stack((dados_fluxo_energia[:, 0], densidades_espaco_fase))
    dados_para_salvar_sem_primeira_coluna = dados_para_salvar[:, 1:]

# Salvar o arquivo sem a primeira coluna
    arquivo_densidades_espaco_fase = '/home/benetti/Documentos/Instabilidade modulacional/Python/densidades_espaco_fase.txt'
    np.savetxt(arquivo_densidades_espaco_fase, dados_para_salvar_sem_primeira_coluna, delimiter='\t')
    
    
    
    # Salve as densidades no espaço de fase em um arquivo TXT
    #arquivo_densidades_espaco_fase = '/home/benetti/Documentos/Instabilidade modulacional/Python/densidades_espaco_fase.txt'
   # np.savetxt(arquivo_densidades_espaco_fase, np.column_stack((dados_fluxo_energia[:, 0], densidades_espaco_fase)),
               #delimiter='\t')

    print(f'Densidades no espaço de fase salvas em {arquivo_densidades_espaco_fase}')
else:
    print(f'A variável {nome_variavel_energia} não foi encontrada no arquivo.')

# Feche o arquivo CDF quando terminar de usá-lo
cdf.close()

arquivo_energia_velocidade = '/home/benetti/Documentos/Instabilidade modulacional/Python/energia_velocidade.txt'
dados_energia_velocidade = np.loadtxt(arquivo_energia_velocidade, delimiter='\t')
colunas_energia_velocidade = ['Energia(eV)', 'Velocidade(10e6m/s)']
df_energia_velocidade = pd.DataFrame(dados_energia_velocidade, columns=colunas_energia_velocidade)

# Carregar os dados de "dados_exportados_angulo.txt"
arquivo_dados_exportados = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_exportados_angulo.txt'
dados_dados_exportados = np.loadtxt(arquivo_dados_exportados, delimiter='\t')
colunas_dados_exportados = [f'Densidade {i + 1}' for i in range(dados_dados_exportados.shape[1])]
df_dados_exportados = pd.DataFrame(dados_dados_exportados, columns=colunas_dados_exportados)

  
# Carregar os dados de "densidades_espaco_fase.txt"
arquivo_densidades_espaco_fase = '/home/benetti/Documentos/Instabilidade modulacional/Python/densidades_espaco_fase.txt'
dados_densidades_espaco_fase = np.loadtxt(arquivo_densidades_espaco_fase, delimiter='\t')
#colunas_densidades_espaco_fase = [f'Densidade{i + 1}' for i in range(dados_densidades_espaco_fase.shape[1])]
colunas_densidades_espaco_fase = tempos_formatados
df_densidades_espaco_fase = pd.DataFrame(dados_densidades_espaco_fase, columns=colunas_densidades_espaco_fase)

# Mesclar os DataFrames usando a posição das linhas como índice e garantir valores NaN
#df_final = pd.merge(df_energia_velocidade, df_dados_exportados, left_index=True, right_index=True, how='outer')
df_final = pd.merge(df_energia_velocidade, df_densidades_espaco_fase, left_index=True, right_index=True, how='outer')

# Substituir valores NaN por "NA" (ou qualquer outra string desejada)
df_final = df_final.fillna("0")

# Salvar o DataFrame em um único arquivo TXT
arquivo_final = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_combinados.txt'
df_final.to_csv(arquivo_final, sep='\t', index=False)

print(f'Dados combinados salvos em {arquivo_final}')


# Carregar o arquivo dados_combinados.txt
arquivo_combinado = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_combinados.txt'
df_combinado = pd.read_csv(arquivo_combinado, sep='\t')

# Excluir as últimas 6 linhas de cada coluna
df_sem_ultimas_linhas = df_combinado.iloc[:-10, :]
#colunas_com_zeros = (df_sem_ultimas_linhas.iloc[1:] == 0).all()

# Excluir colunas com valores iguais a zero a partir da segunda linha
#df_sem_zeros = df_sem_ultimas_linhas.drop(columns=colunas_com_zeros[colunas_com_zeros].index)


# Salvar o novo DataFrame em um arquivo TXT
arquivo_sem_ultimas_linhas = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_sem_ultimas_linhas.txt'
df_sem_ultimas_linhas.to_csv(arquivo_sem_ultimas_linhas, sep='\t', index=False)

print(f'Dados sem as últimas 6 linhas salvos em {arquivo_sem_ultimas_linhas}')



################## fit

caminho_arquivo2 = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_combinadosti2.txt'

# Carregar dados do arquivo
dados_arquivo = np.loadtxt(caminho_arquivo2, delimiter=None, skiprows =1)
if len(dados_arquivo.shape) == 1:
    # Se o array é unidimensional, significa que só tem uma linha
    segunda_linha = dados_arquivo
else:
    # Se o array é bidimensional, obtenha a segunda linha
    segunda_linha = dados_arquivo[1, :]
    
    
# Inicializar uma lista para armazenar os resultados da operação
resultados_operacao = []
resultados_operacao2 = []

arquivo_final = '/home/benetti/Documentos/Instabilidade modulacional/Python/dados_sem_ultimas_linhas.txt'
df_final = pd.read_csv(arquivo_final, sep='\t')

betas = []
valores_Tk = []
valores_Te = []
erros_kappa = []
def gaussiana(v, amp, vpg):
    return amp * np.exp(-((v**2)/vpg)*3.3e-8)

def resultadof2(x, param1, param2, param3):
    return param3 * ((1 + (3.3e-8/((param1)*param2))*(x**2))**(-(param1)-1))

    

# Definição da função de ajuste (distribuição de velocidade kappa)
def kappa_velocity_distribution(x, amp, kappa, beta):
     #return amp * ((1 + (1/(2*kappa - 3))*(6.6e-8/(beta))*(x**2))**(-kappa-1))
    return amp * ((1 + (3.3e-8/((kappa)*beta))*(x**2))**(-(kappa)-1))
    #return amp * (1 + (x**2 / (kappa * beta**2)))**(-kappa-1)

 #Função de resíduo

# Ajuste para cada coluna de densidade# ...

# Crie um DataFrame vazio para armazenar os resultados de todos os ajustes
#df_todos_ajustes = pd.DataFrame()
params_fit_lista = []
# Ajuste para cada coluna de densidade
pdf_filename = '/home/benetti/Documentos/Instabilidade modulacional/Python/ajustes_resultados.pdf'
with PdfPages(pdf_filename) as pdf:
    for i, coluna in enumerate(df_final.columns[2:]):
    #for coluna in df_final.columns[2:]:
        x = df_final['Velocidade(10e6m/s)']
        y = df_final[coluna]
        modelo_gaussiana = Model(gaussiana)
        modelo_gaussiana.set_param_hint('amp', value=5e-12, min=1e-13, max=5e-11)

        modelo_gaussiana.set_param_hint('vpg', value=1e-8, min=1e-7, max=1e-5)
        #pesos1 = x**2
        pesos1 = 1
# Realizar o ajuste usando o modelo LMFit para a função gaussiana
        resultado_gaussiana = modelo_gaussiana.fit(y, v=x, weights=pesos1, method='leastsq')

# Exibir os resultados para a função gaussiana
        #print(resultado_gaussiana.fit_report())
        beta_obtido = resultado_gaussiana.params['vpg'].value 
        beta_obtido2 = resultado_gaussiana.params['vpg'].value 
        betas.append(beta_obtido)
    #y = np.log10(df_final[coluna]+1e-22)
        amp_obtida = resultado_gaussiana.params['amp'].value
        modelo = Model(kappa_velocity_distribution)
    # Parâmetros iniciais para cada ajuste
        #params = modelo.make_params(amp=amp_obtida, kappa=4, beta=2)
        params = modelo.make_params(amp=amp_obtida, kappa=4, beta=beta_obtido2)
        params['amp'].value = amp_obtida
        #params['amp'].min = 1e-14
        params['kappa'].min = 1.55
        params['beta'].value = beta_obtido2
        #params['beta'].min = 1e-8
        params['amp'].vary = False
        #params['amp'].max = 1e-11
        params['kappa'].max = 20
        params['beta'].vary = False
        #params['beta'].max = 1e-5
        
        pesos = (x-5)**10
        #pesos = 1 
        
        
        resultado = modelo.fit(y, params, x=x,weights=pesos, method='leastsq' )
        

        params_fit_lista.append(resultado.params.valuesdict())
        errors_dict = {key + '_stderr': param.stderr for key, param in resultado.params.items()}
        
        #print(resultado.fit_report())
        kappaf = resultado.params['kappa'].value
        betaf = resultado.params['beta'].value
        ampf = resultado.params['amp'].value
        #params_fit_lista = resultado.params.valuesdict()
        #errors_dict = {key + '_stderr': param.stderr for key, param in resultado.params.items()}
        #params_fit_lista.update(errors_dict)
        erro_kappa = resultado.params['kappa'].stderr
        erros_kappa.append(erro_kappa)
        #print(f"Erro de kappa: {erro_kappa}")
        #Tk = (((beta_obtido*1000000)**2 * 9.11e-31)/(1.38e-23))*(2*kappaf/(2*kappaf - 3))
        #Te = ((beta_obtido*1000000)**2 * 9.11e-31)/(1.38e-23)
        #Tk = (((betaf*1000000)**2 * 9.11e-31)/(1.38e-23))*(2*kappaf/(2*kappaf - 3))
        #Te = ((betaf*1000000)**2 * 9.11e-31)/(1.38e-23)
        Tk = beta_obtido2*1e12*(2*kappaf/(2*kappaf - 3))
        Te = beta_obtido*1e12
        valores_Tk.append(Tk)
        valores_Te.append(Te)
        betai = (kappaf-1.5)/(kappaf-0.5)

        

        
    # Visualização dos resultados
        extended_x = np.linspace(-25, 26, 1000)
        extended_y_kappa = kappa_velocity_distribution(extended_x, resultado.params['amp'].value, resultado.params['kappa'].value, beta_obtido2)
        extended_y_gauss = gaussiana(extended_x, resultado_gaussiana.params['amp'].value, resultado_gaussiana.params['vpg'].value)
        plt.yscale('log')
        plt.ylim(1e-20, 1e-11)
        plt.xlim(0, 26)
        
# Ajuste
        plt.plot(x, y, 'bo', label='PSP SPAN-E Data')
        #plt.plot(extended_x, resultado.best_fit, 'r-', label='Ajuste Kappa')
        plt.plot(extended_x, extended_y_kappa, 'r-', label='Kappa')
        plt.title(f'2020 May 28 {coluna} UTC')
        plt.xlabel(r'Velocity ($10^6(ms^{-1})$)')
        plt.ylabel(r'Distribution function ($s^{3}m^{-6}$)')

# Ajuste Gaussiano
        #plt.plot(extended_x, extended_y_kappa, resultado_gaussiana.best_fit, 'g--', label='Ajuste Gaussiano')
        plt.plot(extended_x, extended_y_gauss, 'g--', label='Classical')

# Adicionar legenda
        

# Salvar no arquivo PDF
         
    # Adicionar informações sobre os parâmetros no gráfico
        #info_text = f"Amp: {resultado.params['amp'].value:.2e}\nKappa: {resultado.params['kappa'].value:.2f}\nBeta: {resultado.params['beta'].value:.2f}\nChi Squared: {resultado.chisqr:.2e}"
        info_text = f"Amplitude: {resultado.params['amp'].value:.2e}\n$\\kappa$: {resultado.params['kappa'].value:.2f}\n$\\beta$: {betai:.2f}\n$T_e$: {Te:.2f}\n$T_\kappa$: {Tk:.2f}\nChi Squared: {resultado.chisqr:.2e}"
        
        plt.annotate(info_text, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10)

        

# Valores dos parâmetros
        
        param1 = kappaf  # Substitua pelos seus valores reais
        param2 = beta_obtido  # Substitua pelos seus valores reais
        param3 = ampf
# Calcular os valores de y para cada conjunto de parâmetros
        x_values = np.linspace(0.0, 26, 100)

        

        
        y_values = resultadof2(x_values, param1, param2, param3)
        plt.plot(x_values, y_values,'-r', label = 'Suprathermal')
        plt.legend()
       
        pdf.savefig()  # Salvar o gráfico atual no arquivo PDF
        plt.close()  # Fechar a figura atual para liberar memória
    
        plt.show()
        
        plt.show()
        
segunda_linha = np.where(segunda_linha > 1e8, 0, segunda_linha)
valores_Tk = np.array(valores_Tk)
valores_Te = np.array(valores_Te)
for j in range(len(segunda_linha)):
    # Substitua a fórmula real de Tk aqui
        
    # Operação com o valor correspondente da segunda linha
    #resultadof = 0.5*((1+segunda_linha[j]/valores_Tk[j])/(segunda_linha[j]/valores_Tk[j]))
    #resultadol = 0.5*(1+(valores_Tk[j])/segunda_linha[j])
    resultadol = segunda_linha[j]/valores_Te[j]
    resultadol2 = segunda_linha[j]/valores_Tk[j]
    
    # Adicione o resultado à lista
    resultados_operacao.append(resultadol)
    resultados_operacao2.append(resultadol2)

# Converta a lista de resultados em um array NumPy
resultados_operacao = np.array(resultados_operacao)
resultados_operacao2 = np.array(resultados_operacao2)

#print(f'lang{resultados_operacao}')
#df_ti3 = pd.DataFrame(data=[tempos_formatados,resultados_operacao.T], columns=tempos_formatados)


incerteza=temperaturas_associadas2/valores_Tk 
incerteza = np.where(incerteza > 2, 0, incerteza)

tempos_fit = tempos_formatados  # Use a variável tempos_formatados

amps = [params['amp'] for params in params_fit_lista]
kappas = [params['kappa'] for params in params_fit_lista]
betas = [params['beta'] for params in params_fit_lista]

# Criar um dicionário para o DataFrame
data_dict = {
    'Tempo': tempos_formatados,  # Substitua pelos seus valores reais
    'Ti': segunda_linha,
    'Incerteza': incerteza,
    'Te': valores_Te,
    'Tk': valores_Tk,
    'Ti/Te': resultados_operacao,  # Substitua pelos seus valores reais
    'Ti/Tk': resultados_operacao2,
    'Kappa': kappas,
    'Ik': erros_kappa,
    'Beta': betas,
    'Amp': amps
}
for key, value in data_dict.items():
    print(f"Comprimento de {key}: {len(value)}")

# Criar o DataFrame
df_ti3 = pd.DataFrame(data_dict)





# Extrair os parâmetros ajustados
params_fit = resultado.params.valuesdict()

# Criar um DataFrame com os tempos e parâmetros
df_fit = pd.DataFrame(params_fit_lista, index=tempos_fit).transpose()

#df_ti3 = pd.DataFrame({'Tempo': tempos_formatados, 'Ti/Te': resultados_operacao.tolist()})

    # Salvar o DataFrame em um único arquivo TXT
arquivo_final4 = '/home/benetti/Documentos/Instabilidade modulacional/Python/LWcondi.txt'
#np.savetxt(arquivo_final4, df_ti3, delimiter='\t', fmt='%s')
df_ti3.to_csv(arquivo_final4, sep='\t', index=False, header=['Tempo', 'T_i','Incerteza','Te','Tk','Ti/Te','Ti/Tk','Kappa','Ik', 'Beta', 'Amp' ])

print(f'Dados combinados salvos em {arquivo_final4}')
# Salvar o DataFrame em um arquivo TXT
#arquivo_fit = '/home/benetti/Documentos/Instabilidade modulacional/Python/fit_resultados.txt'
#df_fit.to_csv(arquivo_fit, sep='\t')

#print(f'Parâmetros ajustados salvos em {arquivo_fit}')
#arquivo_fit = '/home/benetti/Documentos/Instabilidade modulacional/Python/fit_resultados.txt'
#df_fit = pd.read_csv(arquivo_fit, sep='\t', index_col=0)

# Extrair os dados de kappa da terceira linha
valores_kappa = df_fit.iloc[1]

# Converter o índice (tempo) para objetos de datetime
tempos = [datetime.strptime(tempo, '%H:%M:%S') for tempo in df_fit.columns]

# Criar gráfico de variação de Kappa ao longo do tempo
plt.figure()
plt.plot(tempos, valores_kappa, 'go-', label='Valor de Kappa')
plt.xlabel('Tempo (horas)')
plt.ylabel('Valor de Kappa')
plt.title('Variação de Kappa ao Longo do Tempo')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend()
plt.show()

def resultadof(x):
    return 0.5 * ((1 + x) / x)

# Calcular os valores de y usando a função resultadof
x_values = np.linspace(0.01, 0.5, 100)

# Calcular os valores de y usando a função resultadof
resultados_y = resultadof(x_values)

# Criar o gráfico
plt.xlim(0.02, 0.5)
plt.ylim(1.5, 20)
plt.plot(x_values, resultados_y, label='resultadof')

arquivo_dados = '/home/benetti/Documentos/Instabilidade modulacional/Python/LWcondi.txt'
df_dados = pd.read_csv(arquivo_dados, sep='\t')

erros_kappa = np.array(erros_kappa)
# Adicionar pontos experimentais ao gráfico
#plt.scatter(df_dados['Ti/Tk'], df_dados['Kappa'], label='Pontos Experimentais', s=3, color='darkred', marker='o')
plt.errorbar(df_dados['Ti/Te'], df_dados['Kappa'], yerr=erros_kappa, linestyle="None", markersize=4, color='darkred', capsize=3, fmt='o', label='Barras de Erro')
#plt.errorbar(df_dados['Ti/Tk'], df_dados['Kappa'], xerr=incerteza, yerr=erros_kappa, linestyle="None", color='red', capsize=3, fmt='o', label='Barras de Erro')
plt.xlabel('Ti/Tk')
plt.ylabel('Kappa')
plt.title('Gráfico de resultadof com Pontos Experimentais')
plt.legend()
plt.show()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
Gráfico



from spacepy import pycdf
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def resultadof(x):
    return 0.5 * ((1+x) / x)

# Calcular os valores de y usando a função resultadof
x_values = np.linspace(0.01, 0.5, 100)
resultados_y = resultadof(x_values)

# Criar o gráfico
fig, ax = plt.subplots()
ax.set_xlim(0.0, 0.5)
ax.set_ylim(1.5, 6)
ax.plot(x_values, resultados_y, color = 'darkorange', label='Modulational Instability Limit')

arquivo_dados = '/home/benetti/Documentos/Instabilidade modulacional/Python/LWcondi.txt'
df_dados = pd.read_csv(arquivo_dados, sep='\t')
df_dados['Ik'] = np.where(df_dados['Ik'] > 3, 1.5, df_dados['Ik'])
#df_dados['Ik'] = np.where(df_dados['Ik'] < 0.3, 0.7, df_dados['Ik'])
df_dados['Kappa'] = np.where(df_dados['Kappa'] < 1.55, 100, df_dados['Kappa'])
df_dados['Kappa'] = np.where(df_dados['Kappa'] > 15, 100, df_dados['Kappa'])
df_dados['Incerteza'] = np.where(df_dados['Incerteza'] > 0.05, 0.03, df_dados['Incerteza'])
df_dados['Ti/Te'] = np.where(df_dados['Ti/Te'] < 0.03, 1, df_dados['Ti/Te'])
#df_dados = df_dadosfil[df_dados['Kappa'] <= 19]
# Adicionar pontos experimentais abaixo da curva ao gráfico
#below_curve = df_dados[df_dados['Kappa'] < resultadof(df_dados['Ti/Tk'])]
below_curve = df_dados[np.where((df_dados['Ti/Te'] != 0) & (df_dados['Ti/Te'] < 0.5), (df_dados['Kappa']-df_dados['Ik']) < resultadof(df_dados['Ti/Te']), False)]
#below_curve = df_dados[np.where((df_dados['Ti/Tk'] != 0) & (df_dados['Ti/Tk'] < 0.5), (df_dados['Kappa']) < resultadof(df_dados['Ti/Tk']), False)]

# Cores distintas para os pontos abaixo da curva
unique_colors = plt.cm.get_cmap('cool')(np.linspace(0, 1, len(below_curve['Tempo'].unique())))

# Adicionar pontos experimentais ao gráfico com cores distintas
ax.errorbar(df_dados['Ti/Te'], df_dados['Kappa'], yerr = df_dados['Ik'], xerr = df_dados['Incerteza'], linestyle="None", markersize=4, color='cornflowerblue', capsize=3,label = 'Stability', fmt='o',  marker='o')

# Adicionar legendas para os pontos abaixo da curva
for i, (index, row) in enumerate(below_curve.iterrows()):
    tempo, ti_te, kappa, Incerteza, Ik = row['Tempo'], row['Ti/Te'], row['Kappa'], row['Incerteza'], row['Ik']
    
    ax.errorbar(ti_te, kappa, yerr = Ik, xerr = Incerteza, linestyle="None", markersize=4,capsize=3, color=unique_colors[i], fmt='o', label=f' Instability- {tempo} - $T_i/T_e$: {ti_te:.2f} - $\kappa$: {kappa:.2f}')

# Adicionar legenda fora do gráfico
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=2)

ax.set_xlabel(r'$T_i/T_e$')
ax.set_ylabel(r'kappa$(\kappa)$')
ax.set_title('PSP 2023 March 23')
plt.savefig('/home/benetti/Documentos/Instabilidade modulacional/Python/MI.pdf', format='pdf', bbox_inches='tight')
plt.show()

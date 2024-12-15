# Importação de bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Propriedades
d = [0, 0.0001, 0.0008, 0.0016, 0.0036, 0.0116]  # Distância de cada interface até a superfície da pele [m]
H = 0.0116  # Tamanho total até a borda interna [m]
p = [1030, 1200, 1200, 1200, 1000, 1085]   # Massas específicas das camadas [kg/m³]
c = [3852, 3589, 3300, 3300, 2674, 3800]    # Calores específicos das camadas [J/kgK]
k = [0.558, 0.235, 0.445, 0.445, 0.185, 0.510]  # Condutividades térmicas das camadas [W/mK]
w_bas = [0.0063, 0.0, 0.0002, 0.0013, 0.0001, 0.0027]   # Taxa de perfusão [1/c]
Q_bas = [3700, 0.0, 368.1, 368.1, 368.3, 684.2] # Produção de calor por volume do metabolismo [W/m³]
q = [1.1, 2.0, 2.0, 2.0, 2.0, 2.0]  # Fatores qb e qm definidos no artigo
pb, cb = 1060, 3700 # Propriedades do sangue
Ta = 37 # Temperatura arterial normal [°C]
Te = 20 # Temperatura ambiente [°C]
h = 100 # Coeficiente de transferência de calor por convecção com o ar [W/m²K]

# Discretização do espaço e tempo
N = 2  # Fator de divisão do espaço [x116 divisões] (inteiro maior que 1)
interfaces = [0,N,8*N,16*N,36*N,116*N]    # Identificação dos nós de interface entre camadas
N *= 116
dx = H / N  # Passo de espaço [m]
dt = 1  # Passo de tempo [s]
nos = np.linspace(0, H, N+1)
temperaturas = np.full(N+1, Ta) # Condições iniciais de temperatura

# Identificação da posição e tamanho do tumor no grid
p_tumor = float(input('Posição do tumor [mm]: '))    # Posição do tumor [m]
tam_tumor = float(input('Tamanho do tumor [mm]: '))   # Tamanho do tumor [m]
Tt = int(input('Tempo total [s]: '))
p_tumor *= 10**-3
tam_tumor *= 10**-3
pi, pf = p_tumor - tam_tumor, p_tumor + tam_tumor
noi, nof = round(pi/dx), round(pf/dx)   # Nós limites do tumor

# Função produção térmica por volume devido ao metabolismo
def Qm(m,T):
    return Q_bas[m]*q[m]**((T-Ta)/10)

# Função taxa de perfusão de sangue
def wm(m,T):
    return w_bas[m]*q[m]**((T-Ta)/10)

# Criação das matrizes
A = np.zeros((N+1,N+1))
B = np.zeros(N+1)

# Função para definir a camada com base no nó
def camada(i):
    if noi < i <= nof:
        return 0
    else:
        if i <= interfaces[1]:
            return 1
        if interfaces[1] < i <= interfaces[2]:
            return 2
        if interfaces[2] < i <= interfaces[3]:
            return 3
        if interfaces[3] < i <= interfaces[4]:
            return 4
        if interfaces[4] < i <= interfaces[5]:
            return 5

# Populando a matriz dos coeficientes
A[0,0] = 3*k[1] + 2*h*dx
A[0,1] = -4*k[1]
A[0,2] = k[1]
for i in range(1, N):
    m = camada(i)
    if (i in interfaces and not noi<i<nof) or i in [noi,nof]:
        A[i,i-2] = k[m]
        A[i,i-1] = -4*k[m]
        A[i,i] = 3*(k[m] + k[camada(i+1)])
        A[i,i+1] = -4*k[camada(i+1)]
        A[i,i+2] = k[camada(i+1)]
    else:
        A[i,i-1] = - k[m] / dx**2
        A[i,i] = 2*k[m] / dx**2 + p[m]*c[m] / dt
        A[i,i+1] = - k[m] / dx**2
A[N,N] = 1

# Populando a matriz dos termos independentes
B[0] = 2*h*dx*Te
B[N] = Ta
def popB(temperaturas):
    for i in range(1,N):
        m = camada(i)
        if i not in interfaces or noi<i<nof:
            Ti = temperaturas[i]
            B[i] = pb*cb*wm(m,Ti)*(Ta - Ti) + Qm(m,Ti) + p[m]*c[m]*Ti / dt

# Criação do gráfico de temperaturas
for t in range(Tt):
    popB(temperaturas)
    temperaturas = np.linalg.solve(A,B)
fig, ax = plt.subplots()
ax.plot(nos,temperaturas)
plt.ylim(min(temperaturas),max(temperaturas))
plt.show()
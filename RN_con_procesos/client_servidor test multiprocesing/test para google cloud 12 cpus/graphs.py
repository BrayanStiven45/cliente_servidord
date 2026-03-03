import pandas as pd
import matplotlib.pyplot as plt
import glob

# ==========================================
# CARGAR ARCHIVOS CORRECTAMENTE
# ==========================================

files = glob.glob("results_*.csv")

# Ordenar por número de procesos (no alfabéticamente)
files = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

print("Archivos encontrados:", files)

data_summary = []

for file in files:
    n_proc = int(file.split("_")[1].split(".")[0])
    
    df = pd.read_csv(file)
    
    data_summary.append({
        "processes": n_proc,
        "mean_time": df["time"].mean(),
        "std_time": df["time"].std(),
        "mean_accuracy": df["accuracy"].mean(),
        "std_accuracy": df["accuracy"].std()
    })

summary_df = pd.DataFrame(data_summary)

print(summary_df)

# ==========================================
# GRAFICA TIEMPO
# ==========================================

plt.figure()
plt.errorbar(summary_df["processes"],
             summary_df["mean_time"],
             yerr=summary_df["std_time"],
             marker='o')

plt.xlabel("Número de procesos")
plt.ylabel("Tiempo promedio (s)")
plt.title("Tiempo vs Número de procesos")
plt.grid(True)
plt.show()
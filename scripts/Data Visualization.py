#Data Visualization
# Pie chart for application type distribution
plt.figure(figsize=(8, 6))
data["application type"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Distribution of Application Types")
plt.show()

# Histograms for numerical features
numerical_features = ["signal strength (dBm)", "latency (ms)", "required bandwidth (Mbps)", "allocated bandwidth (Mbps)", "resource allocation"]
data[numerical_features].hist(figsize=(12, 8), bins=10)
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

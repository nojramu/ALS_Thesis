import numpy as np

import plotly.graph_objects as go
import plotly.express as px

# 1. 3D Surface Plot
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
x, y = np.meshgrid(x, y)
z = np.sin(x ** 2 + y ** 2)

fig1 = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
fig1.update_layout(title='3D Surface Plot', autosize=False, width=500, height=400)

# 2. Sunburst Diagram
data = dict(
    character=["Earth", "Land", "Forest", "Desert", "Water", "Ocean", "Lake"],
    parent=["", "Earth", "Land", "Land", "Earth", "Water", "Water"],
    value=[10, 4, 2, 2, 6, 4, 2]
)
fig2 = px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
    title='Sunburst Diagram'
)

# 3. Scatter Plot with Trendline
df = px.data.iris()
fig3 = px.scatter(
    df, x="sepal_width", y="sepal_length", color="species",
    trendline="ols", title="Iris Sepal Dimensions"
)

# 4. Polar Bar Chart
theta = ['A', 'B', 'C', 'D']
r = [1, 2, 3, 4]
fig4 = go.Figure(data=go.Barpolar(
    r=r,
    theta=theta,
    marker_color=["#FF5733", "#33FFCE", "#335BFF", "#FF33A8"],
    marker_line_color="black",
    marker_line_width=2,
    opacity=0.8
))
fig4.update_layout(title='Polar Bar Chart')

# Show all figures
fig1.show()
fig2.show()
fig3.show()
fig4.show()
def plot_pie(ax, data, column_name, title, threshold=0):
    value_counts = data[column_name].value_counts()
    mask = value_counts.values / len(data) < threshold
    other_count = value_counts[mask].sum()
    value_counts = value_counts[~mask]

    if other_count > 0:
        value_counts["Other"] = other_count

    ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)





# TODO: remove
# def make_pie_plot(data: pd.DataFrame, column_name):
#     value_counts = data[column_name].value_counts()
#     plt.figure(figsize=(6, 6))
#     plt.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
#     plt.title(f"{column_name} Distribution")
#     plt.show()


# def make_pie_plot(data: pd.DataFrame, column_name, threshold=0.05, title="Distribution"):
#     value_counts = data[column_name].value_counts()
#
#     mask = value_counts.values/len(data) < threshold
#     other_count = value_counts[mask].sum()
#     value_counts = value_counts[~mask]
#
#     if other_count > 0:
#         value_counts["Other"] = other_count
#
#     plt.figure(figsize=(6, 6))
#     plt.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
#     plt.title(title)
#     plt.show()


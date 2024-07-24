import matplotlib.pyplot as plt


def plot_pie(ax, data, column_name, title, threshold=0):
    value_counts = data[column_name].value_counts()
    mask = value_counts.values / len(data) < threshold
    other_count = value_counts[mask].sum()
    value_counts = value_counts[~mask]

    if other_count > 0:
        value_counts["Other"] = other_count

    ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)


def plot_draw_percentage_by_base_time(data_df, base_times=None):
    """
    Plots the draw percentage by base time.

    :param data_df: DataFrame with "BaseTime" and "Draw" columns.
    :param base_times: Optional; either a list of specific base times or an integer n for the top n popular base times.
    """
    if base_times is None:
        base_time_df = data_df[["BaseTime", "Draw"]]
    elif isinstance(base_times, int):
        base_time_counts = data_df["BaseTime"].value_counts()
        top_base_times = base_time_counts.head(base_times).index
        base_time_df = data_df[data_df["BaseTime"].isin(top_base_times)][["BaseTime", "Draw"]]
    else:
        base_time_df = data_df[data_df["BaseTime"].isin(base_times)][["BaseTime", "Draw"]]

    pct_draw_values = base_time_df.groupby("BaseTime").mean() * 100

    _, plot_ax = plt.subplots(figsize=(12, 6))
    pct_draw_values.plot(kind="bar", ax=plot_ax)
    plot_ax.set_title("Draw Percentage by Base Time")
    plot_ax.set_ylabel("Draw Percentage (%)")
    plot_ax.set_xlabel("Base Time")
    plt.xticks(rotation=45)
    plt.show()




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


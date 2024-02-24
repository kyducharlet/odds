import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == "__main__":
    experiments = ["synthetics", "conveyors"]
    """ Plot DyCF comparison """
    df_full = pd.DataFrame()
    for expe in experiments:
        df = pd.read_csv(f"results_{expe}_experiment.csv", index_col=0)
        df = df.rename(columns={"Méthode": "Method", "Jeu": "Dataset", "Durée": "Duration", "Taille": "Size"})
        df = df.set_index(["Method", "Dataset", "Index"]).stack().reset_index().rename(columns={"level_3": "Metric", 0: "Value"})
        df_table = pd.DataFrame(columns=df["Method"].unique())
        df_full = pd.concat([df_full, df])
        sns.set_style("darkgrid")
        sns.set_context("paper")
        for dataset in df["Dataset"].unique():
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            plt.suptitle(dataset)
            df_sub = df[df["Dataset"] == dataset].drop(columns=["Dataset"])
            metrics = df_sub[df_sub["Metric"] != "Size"]["Metric"].unique()
            for i in range(3):
                df_sub_metric = df_sub[df_sub["Metric"] == metrics[i]]
                df_table.loc[dataset + f"_mean_{metrics[i]}"] = [df_sub_metric[(df_sub_metric["Method"] == method)]["Value"].mean() for method in df_table.columns]
                df_table.loc[dataset + f"_std_{metrics[i]}"] = [df_sub_metric[df_sub_metric["Method"] == method]["Value"].std() for method in df_table.columns]
                df_sub_metric = df_sub_metric.dropna()
                palette = {
                    "KDE": sns.color_palette("pastel")[5],
                    "SmartSifter": sns.color_palette("pastel")[4],
                    "DBOKDE": sns.color_palette("pastel")[1],
                    "ILOF": sns.color_palette("pastel")[0],
                    "DyCF": sns.color_palette()[3],
                    "DyCG": sns.color_palette()[2],
                }
                bp = sns.barplot(df_sub_metric, x="Metric", y="Value", hue="Method", palette=palette, ax=axes[i])
            axes[0].get_legend().remove()
            axes[1].get_legend().remove()
            sns.move_legend(axes[2], "upper left", bbox_to_anchor=(1, 1))
            plt.savefig(f"results_{expe}_{dataset}.png", bbox_inches="tight")
            plt.close()
        metrics = df[df["Metric"] != "Size"]["Metric"].unique()
        for i in range(3):
            df_table.loc["global" + f"_mean_{metrics[i]}"] = [df[(df["Method"] == method) & (df["Metric"] == metrics[i])]["Value"].mean() for method in df_table.columns]
            df_table.loc["global" + f"_std_{metrics[i]}"] = [df[(df["Method"] == method) & (df["Metric"] == metrics[i])]["Value"].std() for method in df_table.columns]
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('off')
        pd.plotting.table(ax, df_table, loc='center', cellLoc='center')
        plt.savefig(f"results_{expe}_table.png", bbox_inches="tight")
        plt.close()

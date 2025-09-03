import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def summarize_targets(y_raw: np.ndarray, y_imp: np.ndarray, target_names):
    """
    Print descriptive stats of targets before and after imputation safely.
    """
    try:
        assert y_raw.shape == y_imp.shape
        for j, name in enumerate(target_names):
            print(f"\n=== {name} ===")
            try:
                orig = y_raw[:, j]
                imp  = y_imp[:, j]
                mask_known   = ~np.isnan(orig)
                mask_missing =  np.isnan(orig)

                # Known values
                print(f"Known (before imputation): n={mask_known.sum()}")
                if mask_known.any():
                    print(pd.Series(orig[mask_known]).describe())

                # Imputed values
                print(f"\nImputed count: {mask_missing.sum()}")
                if mask_missing.any():
                    print("Imputed-only stats:")
                    print(pd.Series(imp[mask_missing]).describe())

                # Full
                print("\nAfter imputation (full vector) stats:")
                print(pd.Series(imp).describe())
            except Exception as e:
                print(f"Error summarizing {name}: {e}")
    except Exception as e:
        print(f"safe_summarize_targets() failed: {e}")


def plot_target_distributions(y_raw: np.ndarray, y_imp: np.ndarray, target_names,
                                   bins=30, figsize=(10, 4)):
    """
    Plots distribution of known vs. imputed values per target with error handling.
    """
    try:
        assert y_raw.shape == y_imp.shape
        n_targets = y_raw.shape[1]

        for j, name in enumerate(target_names):
            try:
                orig = y_raw[:, j]
                imp  = y_imp[:, j]

                known_mask   = ~np.isnan(orig)
                missing_mask =  np.isnan(orig)

                fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
                fig.suptitle(f"{name} — distributions before & after imputation", fontsize=12)

                if known_mask.any():
                    ax[0].hist(orig[known_mask], bins=bins)
                ax[0].set_title(f"{name} (known only)\n n={known_mask.sum()}")
                ax[0].set_xlabel(name)
                ax[0].set_ylabel("Count")

                if missing_mask.any():
                    ax[1].hist(imp[missing_mask], bins=bins)
                ax[1].set_title(f"{name} (imputed only)\n n={missing_mask.sum()}")
                ax[1].set_xlabel(name)
                ax[1].set_ylabel("Count")

                plt.show()

            except Exception as e:
                print(f"Error plotting {name}: {e}")
    except Exception as e:
        print(f"safe_plot_target_distributions() failed: {e}")


def plot_overlay_pre_post(y_raw: np.ndarray, y_imp: np.ndarray, target_names,
                               bins=30, alpha_known=0.6, alpha_imp=0.6, figsize=(6,4)):
    """
    Overlay known vs. imputed distributions for each target.
    """
    try:
        for j, name in enumerate(target_names):
            try:
                orig = y_raw[:, j]
                imp  = y_imp[:, j]
                known_mask   = ~np.isnan(orig)
                missing_mask =  np.isnan(orig)

                plt.figure(figsize=figsize)
                if known_mask.any():
                    plt.hist(orig[known_mask], bins=bins, alpha=alpha_known,
                             label="Known (pre)", density=True)
                if missing_mask.any():
                    plt.hist(imp[missing_mask], bins=bins, alpha=alpha_imp,
                             label="Imputed-only (post)", density=True)
                plt.title(f"{name} — overlay distributions")
                plt.xlabel(name)
                plt.ylabel("Density")
                plt.legend()
                plt.show()

            except Exception as e:
                print(f"Error in overlay plot for {name}: {e}")
    except Exception as e:
        print(f"safe_plot_overlay_pre_post() failed: {e}")
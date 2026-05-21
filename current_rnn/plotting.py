import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import torch as tch
import numpy as np

def plot_loss(epoch, losses, title, tag = ''):
    plt.figure()
    plt.semilogy(range(epoch), losses[:epoch])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(tag + '.png',  bbox_inches = 'tight')
    plt.close()


def plot_training_history_summary(df, out_png, title=None, train_stride=None):
    if df is None or len(df) == 0:
        return

    n_rows = int(len(df))
    if train_stride is None:
        train_stride = max(1, int(np.ceil(float(n_rows) / 5000.0)))
    train_stride = max(1, int(train_stride))

    df_train = df.iloc[::train_stride, :].copy() if train_stride > 1 else df.copy()
    epochs_train = df_train["epoch"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    train_cols = [
        ("train_total", "train_total"),
        ("train_psth", "train_psth"),
        ("train_psth_weighted", "train_psth_weighted"),
        ("train_psth_sample", "train_psth_sample"),
        ("train_psth_delay", "train_psth_delay"),
        ("train_psth_response", "train_psth_response"),
        ("train_type", "train_type"),
        ("train_J", "train_J"),
    ]
    for col, label in train_cols:
        if col in df_train.columns:
            y = df_train[col].to_numpy(dtype=float)
            if np.isfinite(y).any():
                axes[0].plot(epochs_train, y, label=label, linewidth=1.2)
    axes[0].set_title("Train losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels)

    eval_mask = np.zeros(len(df), dtype=bool)
    for col in ("eval_total", "eval_psth", "eval_type", "eval_J"):
        if col in df.columns:
            eval_mask |= np.isfinite(df[col].to_numpy(dtype=float))
    df_eval = df.loc[eval_mask].copy()
    if len(df_eval) > 0:
        epochs_eval = df_eval["epoch"].to_numpy(dtype=float)
        eval_cols = [
            ("eval_total", "eval_total"),
            ("eval_psth", "eval_psth"),
            ("eval_psth_sample", "eval_psth_sample"),
            ("eval_psth_delay", "eval_psth_delay"),
            ("eval_psth_response", "eval_psth_response"),
            ("eval_type", "eval_type"),
            ("eval_J", "eval_J"),
        ]
        for col, label in eval_cols:
            if col in df_eval.columns:
                y = df_eval[col].to_numpy(dtype=float)
                if np.isfinite(y).any():
                    axes[1].plot(epochs_eval, y, label=label, linewidth=1.2)
    axes[1].set_title("Eval metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels)

    j_cols = [
        ("J_frob", "J_frob"),
        ("J_maxabs", "J_maxabs"),
    ]
    for col, label in j_cols:
        if col in df_train.columns:
            y = df_train[col].to_numpy(dtype=float)
            if np.isfinite(y).any():
                axes[2].plot(epochs_train, y, label=label, linewidth=1.2)
    axes[2].set_title("J stats")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("J stats")

    ax2b = axes[2].twinx()
    if "lambda_J_eff" in df_train.columns:
        y = df_train["lambda_J_eff"].to_numpy(dtype=float)
        if np.isfinite(y).any():
            ax2b.plot(epochs_train, y, label="lambda_J_eff", linestyle="--", color="black", linewidth=1.2)
    ax2b.set_ylabel("lambda_J_eff")

    handles_l, labels_l = axes[2].get_legend_handles_labels()
    handles_r, labels_r = ax2b.get_legend_handles_labels()
    if handles_l or handles_r:
        axes[2].legend(handles_l + handles_r, labels_l + labels_r, loc="best")

    if title:
        fig.suptitle(str(title))
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_neurons(net, rates, default_parameters, t_del = -.03, tag = ''):    
    with tch.set_grad_enabled(False):
        y_alm_l = rates['ALM_l']
        y_alm_r = rates['ALM_r']
        input_net = {'ALM_l':y_alm_l}
        results, _ = net(input_net)
        rates_alm_l = results['rates_alm_l']
        rates_alm_r = results['rates_alm_r']
        rates_octx_l = results['rates_octx_l']
        rates_octx_r = results['rates_octx_r']
        rates_alm_l = rates_alm_l.detach().cpu().numpy()
        rates_alm_r = rates_alm_r.detach().cpu().numpy()
        rates_octx_l = rates_octx_l.detach().cpu().numpy()
        rates_octx_r = rates_octx_r.detach().cpu().numpy()
        y_alm_l = y_alm_l.detach().cpu().numpy()
        y_alm_r = y_alm_r.detach().cpu().numpy()
        t_start = default_parameters['t_start']
        dt = default_parameters['dt']
        n_time = y_alm_l.shape[1]
        x = np.linspace(0, n_time, n_time) * dt
        x = x + t_start
        n_neurons = 1000
        for j in range(0, n_neurons,10):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_alm_l[0, :, j], 'r--',  lw =3) 
            plt.plot(x, y_alm_l[0, :, j], 'r-', lw =3) 
            plt.plot(x, rates_alm_l[1, :, j], 'b--',  lw =3) 
            plt.plot(x, y_alm_l[1, :, j], 'b-', lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM L")
            plt.savefig('out/neuronal_dynamics/neural_ALM_L_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()
        for j in range(0, n_neurons,10):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_alm_r[0, :, j], 'r--',  lw =3) 
            plt.plot(x, y_alm_r[0, :, j], 'r-', lw =3) 
            plt.plot(x, rates_alm_r[1, :, j], 'b--',  lw =3) 
            plt.plot(x, y_alm_r[1, :, j], 'b-', lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title('ALM R')
            plt.savefig('out/neuronal_dynamics/neural_ALM_R_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()
        for j in range(0, 200,20):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_octx_l[0, :, j], 'r-',  lw =3) 
            plt.plot(x, rates_octx_l[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM L")
            plt.savefig('out/neuronal_dynamics/neural_octx_L_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()
        for j in range(0, 200,20):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, rates_octx_r[0, :, j], 'r-',  lw =3) 
            plt.plot(x, rates_octx_r[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title('ALM R')
            plt.savefig('out/neuronal_dynamics/neural_octx_R_'+ tag + '{}.png'.format(j),  bbox_inches = 'tight')
            plt.close()


def plot_latents(net,  rates, default_parameters, t_del = -.03, tag = ''):   
    with tch.set_grad_enabled(False):
        y_alm_l = rates['ALM_l']
        input_net = {'ALM_l':y_alm_l}
        results, _ = net(input_net)
        latents_alm_l = results['latents_alm_l']
        latents_alm_r = results['latents_alm_r']
        latents_alm_l = latents_alm_l.detach().cpu().numpy()
        latents_alm_r = latents_alm_r.detach().cpu().numpy()
     
   
        t_start = default_parameters['t_start']
        dt = default_parameters['dt']
        n_time = y_alm_l.shape[1]
        x = np.linspace(0, n_time, n_time) * dt
        x = x + t_start
        for j in range(latents_alm_l.shape[2]):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, latents_alm_l[0, :, j], 'r-',  lw =3) 
            plt.plot(x, latents_alm_l[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM L")
            plt.savefig('out/latents/latents_ALM_L_'+tag + '{}.png'.format(j), bbox_inches = 'tight')
            plt.close()
        for j in range(latents_alm_r.shape[2]):
            plt.figure(figsize=(6, 3))
            plt.axvline(x=net.t_min_input + t_del, color='gray', linestyle='--')
            plt.axvline(x=net.t_go + t_del, color='gray', linestyle='--') 
            plt.axvline(x=net.t_max_input + t_del, color='gray', linestyle='--')
            plt.plot(x, latents_alm_r[0, :, j], 'r-',  lw =3) 
            plt.plot(x, latents_alm_r[1, :, j], 'b-',  lw =3) 
            plt.xlabel('Time (s)')
            plt.ylabel('Mode value')
            plt.title("ALM R")
            plt.savefig('out/latents/latents_ALM_R_'+tag + '{}.png'.format(j), bbox_inches = 'tight')
            plt.close()

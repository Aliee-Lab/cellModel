import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import os
from .PARM_utils_load_model import load_PARM
from .PARM_misc import log, check_sequence_length
from tqdm import tqdm


def PARM_predict(
    input: str,
    output: str,
    model_directory: str,
    n_seqs_per_batch: int = 1,
    store_sequence: bool = True,
    filter_size: int = 125,
    type_loss: str = "poisson",
    test_fold: bool = False,
    L_max: int = 600,
    cell_type: str = None,
):
    """
    Reads the input (fasta file) and predicts promoter activity scores using the PARM models.
    Writes the output as tab-separated values, where each column is a model and each row is a sequence.

    Parameters
    ----------
    input : str
        Path to the input fasta file.
    output : str
        Path to the output file.
    model_directory : str
        Path to the directory containing all the folds of the PARM models. (e.g. K562/* , with K562/K562_fold1.parm, K562_fold2.parm, etc.)
    n_seqs_per_batch : int
        Number of batches to use for prediction. If your GPUs runs out of memory, might be because of that. Default is 1.
    store_sequence : bool
        If True, the output file will contain the sequences and headers. Otherwise, only the headers and score will be saved. Default is True.
    filter_size : int
        Size of the filter to use for the PARM model. Default is 125.
    type_loss : str
        Type of loss function to use for the model. Default is 'poisson'. Other options are 'MSE' and 'heteroscedastic'.
    test_fold : bool
        If True, the function will consider that the input is a hdf5 file with the test fold data. Default is False.
    cell_type : str
        Only needed for multi-output models (trained with more than one cell type):
        the same '__'-joined cell type string used at training time (e.g.
        "HepG2__K562__A549"), in the SAME order the model's output columns were
        trained in. Used only to label output columns; ignored for single-output
        models. If omitted for a multi-output model, columns are labeled
        "output0", "output1", etc. instead.
    -------
    None

    Examples
    --------
    >>> PARM_predict("input.fasta", "output.tsv", ["model1.parm", "model2.parm"])
    """
    # Load models
    log("Loading models")
    list_of_models = list()
    # Iterate over the model_directory and get the folds
    path_to_all_folds = []
    for file in os.listdir(model_directory):
        if file.endswith(".parm"):
            path_to_all_folds.append(os.path.join(model_directory, file))
    if len(path_to_all_folds) == 0:
        raise ValueError(
            f"No model files (.parm) found in {model_directory}. Please check the path and ensure it contains the model files."
        )
    # Now, load the models
    log(f"Found {len(path_to_all_folds)} model files in {model_directory}")
    model_name = ""
    for fold_path in path_to_all_folds:
        if model_name == "":
            model_name = os.path.basename(fold_path).split("_fold")[0]
        elif model_name != os.path.basename(fold_path).split("_fold")[0]:
            raise ValueError(
                f"Model prefixes do not match: {model_name} and {os.path.basename(fold_path).split('_fold')[0]}. Please make sure that folds of the same model have the same prefix"
            )

        # load_PARM needs to construct linear1 with the RIGHT output shape before
        # load_state_dict runs, but a checkpoint doesn't self-describe that count
        # anywhere accessible except inside its own tensors -- so peek at the
        # checkpoint's linear1.weight shape first, rather than assuming n_outputs=1
        # (the default) and having load_state_dict fail with a shape mismatch.
        checkpoint_state = torch.load(fold_path, map_location="cpu")
        n_outputs_in_checkpoint = checkpoint_state["linear1.weight"].shape[0]

        if cell_type is not None:
            if len(cell_type.split("__")) != n_outputs_in_checkpoint:
                raise ValueError(
                    f"--cell_type gives {len(cell_type.split('__'))} name(s) but the "
                    f"checkpoint {fold_path} has {n_outputs_in_checkpoint} output(s). "
                    f"They must match, in training order."
                )
            cell_line_for_loading = cell_type
        elif n_outputs_in_checkpoint > 1:
            # No --cell_type given but the checkpoint is multi-output: build a
            # placeholder cell_line string just so load_PARM constructs the right
            # SHAPE. Real names for column labeling get filled in below via
            # output_names, once we fall back to output0/output1/... naming.
            cell_line_for_loading = "__".join(str(i) for i in range(n_outputs_in_checkpoint))
        else:
            cell_line_for_loading = False  # single-output, original default behaviour

        list_of_models.append(
            load_PARM(
                fold_path, filter_size=filter_size, train=False, type_loss=type_loss,
                cell_line=cell_line_for_loading,
            )
        )

    n_outputs = list_of_models[0].linear1.out_features
    if cell_type is not None:
        output_names = cell_type.split('__')
        if len(output_names) != n_outputs:
            raise ValueError(
                f"--cell_type gives {len(output_names)} name(s) ({output_names}) but this "
                f"model has {n_outputs} output(s). They must match exactly, in the same "
                f"order used at training time."
            )
    else:
        output_names = [f"output{i}" for i in range(n_outputs)] if n_outputs > 1 else None

    if test_fold:
        log("Performing test fold predictions")
        # Use the input path as the test fold HDF5 file path
        test_fold_path = input

        # Create output directory for test results if it doesn't exist
        test_output_dir = output

        get_test_fold_predictions(
            test_fold_path=test_fold_path,
            list_of_models=list_of_models,
            cell_type=output_names if output_names is not None else model_name,
            output_directory=test_output_dir,
        )
        return
    # Default behaviour: input is a fasta file with sequences to predict
    # Iterate over sequences and predict scores
    # Check input fasta
    check_sequence_length(input, L_max)
    log("Making predictions")
    total_sequences = sum(1 for _ in SeqIO.parse(input, "fasta"))
    log(f"Total sequences: {total_sequences}")
    total_interactions = total_sequences * len(list_of_models)
    pbar = tqdm(total=int(total_interactions / n_seqs_per_batch), ncols=80)

    i = 0
    for i_record, record in enumerate(SeqIO.parse(input, "fasta")):
        # Initiate output df
        sequence = str(record.seq).upper()
        if i == 0:
            tmp = pd.DataFrame({"sequence": [sequence], "header": [record.id]})
        else:
            tmp = pd.concat(
                [tmp, pd.DataFrame({"sequence": [sequence], "header": [record.id]})]
            )

        # Get predictions for all models
        if (i + 1) == n_seqs_per_batch or (i_record == (total_sequences - 1)):
            predictions_all_folds = []
            for model in list_of_models:
                predictions_all_folds.append(
                    get_prediction(tmp.sequence.to_list(), model, L_max=L_max)
                )
                pbar.update(1)
            # Average across folds first: shape (batch, n_outputs)
            avg_predictions = np.mean(predictions_all_folds, axis=0)
            if output_names is None:
                # Single-output model: unchanged behaviour.
                tmp["prediction_" + model_name] = avg_predictions.flatten()
            else:
                # Multi-output model: one column per cell type, in training order.
                for j, name in enumerate(output_names):
                    tmp[f"prediction_{model_name}_{name}"] = avg_predictions[:, j]

            # Store in output df
            # IF it's the first batch, save the df with headers, otherwise, save only the scores
            if i_record < n_seqs_per_batch:
                if store_sequence:
                    tmp.to_csv(output, sep="\t", index=False)
                else:
                    (tmp.drop(columns=["sequence"])).to_csv(
                        output, sep="\t", index=False
                    )
            else:
                if store_sequence:
                    tmp.to_csv(output, sep="\t", index=False, mode="a", header=False)
                else:
                    (tmp.drop(columns=["sequence"])).to_csv(
                        output, sep="\t", index=False, mode="a", header=False
                    )

            i = 0

        else:
            i += 1
    # Write output
    pbar.close()
    log("Finish output file")


def get_test_fold_predictions(
    test_fold_path, list_of_models, cell_type, output_directory
):
    """
    Perform predictions on test fold data and create measured vs predicted plots.

    `cell_type` may be a single string (single-output model, original behaviour) or
    a list of strings (multi-output model, one name per output column, in the SAME
    order the model was trained on).
    """
    import h5py

    cell_type_list = [cell_type] if isinstance(cell_type, str) else list(cell_type)
    n_outputs = list_of_models[0].linear1.out_features
    if len(cell_type_list) != n_outputs:
        raise ValueError(
            f"Got {len(cell_type_list)} cell type name(s) ({cell_type_list}) but this "
            f"model has {n_outputs} output(s). They must match, in training order."
        )

    # make output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log(f"Loading test fold data from {test_fold_path}")

    # Load HDF5 file directly
    with h5py.File(test_fold_path, "r") as f:
        # Load sequences (one-hot encoded)
        sequences = f["X"]["sequence"]["OneHotEncoding"][:]
        # Load measured values: one column per cell type -> shape (n_fragments, n_outputs)
        measured = np.stack(
            [np.asarray(f["Y"][f"Log2RPM_{ct}"][:]) for ct in cell_type_list], axis=1
        )
        # Load the feature names of each fragment
        try: 
            feature_names = f["FEAT"]["FEATname"][:].astype(str)
        except: 
            feat_type = f["FEAT/FEATtype"][:].astype(str)
            feat_start = f["FEAT/FEATstart"][:].astype(str)
            feat_end = f["FEAT/FEATend"][:].astype(str)
            feature_names = [
                f"{t}_{s}_{e}"
                for t, s, e in zip(feat_type, feat_start, feat_end)
            ]
    feature_names = np.asarray(feature_names)

    log(f"Loaded {len(sequences)} test fragments")

    # Make predictions with each model
    all_predictions = []
    batch_size = 32
    for i, model in enumerate(list_of_models):
        log(f"Making predictions with model fold {i}")
        model.eval()
        predictions = []

        with torch.no_grad():
            for start_idx in range(0, len(sequences), batch_size):
                end_idx = min(start_idx + batch_size, len(sequences))
                batch_sequences = sequences[start_idx:end_idx]

                # Convert to tensor and move to GPU if available
                X = torch.tensor(batch_sequences, dtype=torch.float32).permute(0, 2, 1)
                if torch.cuda.is_available():
                    X = X.cuda()
                    model = model.cuda()

                # Make predictions -- shape (batch, n_outputs)
                pred = model(X).cpu().detach().numpy()
                predictions.append(pred)

        # Concatenate all batch predictions
        model_predictions = np.concatenate(predictions, axis=0)
        all_predictions.append(model_predictions)

    # Average predictions across all fold models -- shape (n_fragments, n_outputs)
    avg_predictions = np.mean(all_predictions, axis=0)

    # Evaluate + plot each cell type separately -- combining them into one flattened
    # plot (as the original single-output code did) would mix different cell types'
    # scales together and hide any one cell type performing worse than the others.
    for j, ct in enumerate(cell_type_list):
        _evaluate_and_plot_one_celltype(
            predicted=avg_predictions[:, j],
            measured=measured[:, j],
            feature_names=feature_names,
            cell_type=ct,
            output_directory=output_directory,
        )


def _evaluate_and_plot_one_celltype(predicted, measured, feature_names, cell_type, output_directory):
    """The original single-cell-type evaluation body (correlation stats, scatter
    plots, .tsv output), factored out so it runs identically whether there's one
    cell type or several."""
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from scipy.stats import pearsonr

    # h5_dataset-style missing-cell-type entries would show up as NaN here; drop them
    # rather than letting them propagate into every downstream stat as NaN.
    valid = ~np.isnan(measured)
    predicted, measured, feature_names = predicted[valid], measured[valid], feature_names[valid]

    results_df = pd.DataFrame(
        {
            "measured_Log2PM": measured,
            "predicted_Log2RPM": predicted,
            "feature": feature_names,
        }
    )

    # Calculate correlation
    pearson_r, _ = pearsonr(measured, predicted)

    log(f"[{cell_type}] Pearson correlation (fragment level): {pearson_r:.3f}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 7))

    h = ax.hist2d(
        predicted, measured, bins=100, norm=colors.LogNorm(), cmap="viridis"
    )

    # Add correlation annotation
    ax.annotate(
        f"R = {pearson_r:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add diagonal line
    min_val = min(predicted.min(), measured.min())
    max_val = max(predicted.max(), measured.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)

    ax.set_xlabel("Predicted Log2RPM", fontsize=12)
    ax.set_ylabel("Measured Log2RPM", fontsize=12)
    ax.set_title(f"Test Fold Results - {cell_type}", fontsize=14)

    plt.colorbar(h[3], ax=ax, label="Fragment count")

    # Save plot
    plot_path = os.path.join(
        output_directory, f"test_fold_scatter_{cell_type}_fragment_level.svg"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    results_path = os.path.join(
        output_directory, f"test_fold_predictions_{cell_type}.tsv"
    )
    results_df.to_csv(results_path, sep="\t", index=False)

    # Now, group prediction by feature and plot the measured vs. predicted values
    grouped_results = (
        results_df.groupby("feature")
        .agg({"measured_Log2PM": "mean", "predicted_Log2RPM": "mean"})
        .reset_index()
    )

    # plot the grouped results
    fig, ax = plt.subplots(figsize=(8, 7))
    h = ax.hist2d(
        grouped_results["predicted_Log2RPM"],
        grouped_results["measured_Log2PM"],
        bins=100,
        norm=colors.LogNorm(),
        cmap="viridis",
    )
    # Add correlation annotation
    pearson_r_grouped, _ = pearsonr(
        grouped_results["measured_Log2PM"], grouped_results["predicted_Log2RPM"]
    )
    
    log(f"[{cell_type}] Pearson correlation (feature level): {pearson_r_grouped:.3f}")
    
    ax.annotate(
        f"R = {pearson_r_grouped:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    # Add diagonal line
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)
    ax.set_xlabel("Predicted Log2RPM", fontsize=12)
    ax.set_ylabel("Measured Log2RPM", fontsize=12)
    ax.set_title(f"Test Fold Results - {cell_type} (Grouped by Feature)", fontsize=14)
    plt.colorbar(h[3], ax=ax, label="Feature count")
    # Save plot
    plot_grouped_path = os.path.join(
        output_directory, f"test_fold_scatter_{cell_type}_feature_level.svg"
    )
    plt.savefig(plot_grouped_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_prediction(sequence, complete_model, L_max=600):
    """
    Predicts promoter activity score for input sequence
    """
    # Check if sequence is a list or not and make it a list if not
    if not isinstance(sequence, list):
        sequence = [sequence]

    if torch.cuda.is_available():
        complete_model = complete_model.cuda()
    onehot_fragment = torch.tensor(
        np.float32(sequence_to_onehot(sequence, L_max=L_max))
    ).permute(0, 2, 1)
    if torch.cuda.is_available():
        onehot_fragment = onehot_fragment.cuda()
    score = complete_model(onehot_fragment).cpu().detach().numpy()

    return score


def sequence_to_onehot(sequences, L_max, for_mutagenesis=False):
    """
    Transform list of sequences to one hot. Padding is done in the middle, and the padding value is 0.
    Args:
        sequences: (list) of string sequences
        L_max: (int) Max length of sequences. Relevant for padding.

    Returns:
        X_OneHot: (np.array) Array with length (samples, L_max, 4)
    """
    # Define nucleotide to vector
    letter_to_vector = {
        "A": np.array([1.0, 0.0, 0.0, 0.0]),
        "C": np.array([0.0, 1.0, 0.0, 0.0]),
        "G": np.array([0.0, 0.0, 1.0, 0.0]),
        "T": np.array([0.0, 0.0, 0.0, 1.0]),
        "N": np.array([0.0, 0.0, 0.0, 0.0]),
    }

    # get On Hot Encoding
    if for_mutagenesis is False:
        one_hot = []
        for seq in sequences:
            x = np.array([letter_to_vector[s] for s in seq])
            pw = (L_max - x.shape[0]) / 2
            PW = [int(np.ceil(pw)), int(np.floor(pw))]
            one_hot.append(np.pad(x, [PW, [0, 0]], constant_values=0))
        one_hot = np.array(one_hot)
    else:
        # for mutagenesis, the L_max is not set to 600, but is the sequence length, instead
        one_hot = []
        for seq in sequences:
            this_seq_length = len(seq)
            x = np.array([letter_to_vector[s] for s in seq])
            pw = (this_seq_length - x.shape[0]) / 2
            PW = [int(np.ceil(pw)), int(np.floor(pw))]
            one_hot.append(np.pad(x, [PW, [0, 0]], constant_values=0))
        one_hot = np.array(one_hot)

    return one_hot
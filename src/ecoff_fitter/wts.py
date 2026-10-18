import pandas as pd
import multiprocessing as mp
import sys
from joblib import Memory
import piezo


memory = Memory(location=".piezo_cache", verbose=0)
memory.clear(warn=False)

_catalogue_cache = {}


def get_catalogue(catalogue_path):
    """Load and cache the catalogue once per process."""
    global _catalogue_cache
    if catalogue_path not in _catalogue_cache:
        _catalogue_cache[catalogue_path] = piezo.ResistanceCatalogue(catalogue_path)
    return _catalogue_cache[catalogue_path]


class gWTBase:

    @staticmethod
    def parallel_antibiogram(mutations, drug_genes, catalogue_path, cores=4):
        mutations = mutations.set_index("UNIQUEID")
        mut_by_uid = {
            uid: df.reset_index() for uid, df in mutations.groupby("UNIQUEID")
        }

        tasks = []
        for uid in mutations.index.unique():
            iso_muts = mut_by_uid.get(uid, pd.DataFrame(columns=mutations.columns))
            tasks.append((uid, iso_muts, drug_genes, catalogue_path))

        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")

        with ctx.Pool(min(cores, len(tasks))) as pool:
            results = list(
                pool.imap_unordered(gWTBase.process_antibiogram, tasks, chunksize=10),
            )

        antibiograms = {uid: calls for uid, calls in results}

        return antibiograms

    @staticmethod
    def process_antibiogram(args):
        """Generate an antibiogram for one sample"""
        uid, iso_muts, drug_genes, catalogue_path = args
        results = []

        for drug, genes in drug_genes.items():
            muts = iso_muts[iso_muts.GENE.isin(genes)]

            if muts.empty:
                result = "S"
            else:
                preds = muts["MUTATION"].map(
                    lambda var: (
                        "S"
                        if pd.isna(var)
                        else gWTBase.cached_predict(var, drug, catalogue_path)
                    )
                )
                result = (
                    "R"
                    if "R" in preds.values
                    else ("U" if "U" in preds.values else "S")
                )

            results.append(result)

        return (uid, results)

    @staticmethod
    @memory.cache
    def cached_predict(mutation, drug, catalogue_path):
        catalogue = get_catalogue(catalogue_path)
        result = catalogue.predict(mutation)
        return result.get(drug, "S") if isinstance(result, dict) else result


class nonsilent_WT(gWTBase):

    def identify(self, df):

        synonymous_ids, wt_ids = set(), set()

        # Group by 'UNIQUEID' to check mutations
        for unique_id, group in df.groupby("UNIQUEID"):
            mutations = group.MUTATION.dropna()
            if mutations.empty:  # No mutations indicate wild-type
                wt_ids.add(unique_id)
            elif all(m.split("@")[-1][0] == m.split("@")[-1][-1] for m in mutations):
                synonymous_ids.add(unique_id)  # All mutations are synonymous/silent

        # Mark as mutant if not in wild-type or synonymous sets
        return df["UNIQUEID"].isin(synonymous_ids | wt_ids)


class erj2022_WT(gWTBase):

    drugs = ["RIF", "INH", "EMB", "PZA", "AMI", "KAN", "LEV", "MXF", "ETH"]

    def identify(self, mutations, cat_path):

        inf_muts, drug_genes = self.inf_vars(mutations, cat_path)

        antibiograms = self.parallel_antibiogram(
            mutations=inf_muts,
            drug_genes=drug_genes,
            catalogue_path=cat_path,
            cores=mp.cpu_count(),
        )

        first_lines = slice(0, 4)
        second_lines = slice(4, 9)

        gWT_samples = {
            uid: (
                all(pred == "S" for pred in results[first_lines])
                and all(pred in {"S", "U"} for pred in results[second_lines])
            )
            for uid, results in antibiograms.items()
        }

        return gWT_samples

    def inf_vars(self, mutations, cat_path):

        cat = pd.read_csv(cat_path)

        cat["GENE"] = cat["MUTATION"].str.split("@", n=1).str[0]
        R_genes = cat.loc[cat["PREDICTION"].eq("R"), "GENE"].unique()
        cat_R = cat[cat["GENE"].isin(R_genes)]

        # Build mapping: drug -> unique resistant genes
        drug_genes = {
            drug: cat_R.loc[cat_R["DRUG"].eq(drug), "GENE"].unique()
            for drug in self.drugs
        }

        # filter mutations for those in relevant (R) genes
        mutations = mutations[mutations["GENE"].isin(R_genes)]

        return mutations, drug_genes

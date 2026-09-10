"""A ``DataFrameDescriptorSet`` whose join tolerates the PCM row structure.

In a PCM dataset each row is a (compound, target) interaction, so both the compound
and the target identifiers repeat across rows. QSPRpred's ``DataFrameDescriptorSet``
joins its precalculated frame with ``validate="one_to_one"``, which rejects exactly
that: the descriptor frame has unique keys, but the table being described does not.
The join is many-to-one because a target may appear in multiple (compount, target)
interactions.

``many_to_one`` still enforces that the descriptor frame has no duplicate keys.
We provide this subclass because the base class would fail the ``one_to_one`` validation
in ``processMols``.
"""

from typing import Any

import numpy as np
import pandas as pd
from qsprpred.data.descriptors.sets import DataFrameDescriptorSet
from rdkit.Chem import Mol


class PCMDataFrameDescriptorSet(DataFrameDescriptorSet):
    """``DataFrameDescriptorSet`` that broadcasts descriptors over repeated keys."""

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        index_cols = self.getIndexCols()
        if index_cols:
            ret = pd.DataFrame({col: props[col] for col in index_cols})
            ret = self.setIndex(ret, index_cols)
            ret.drop(columns=index_cols, inplace=True)
        else:
            ret = pd.DataFrame(index=pd.Index(props[self.idProp], name=self.idProp))
        ret = ret.join(
            self._df,
            how="left",
            on=index_cols,
            validate="many_to_one",
        )
        return ret[self.descriptors].values

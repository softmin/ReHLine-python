from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        'dataset': ["news20.binary"],
    }

    install_cmd = 'conda'
    requirements = ['pip:libsvmdata']

    def __init__(self, dataset="news20.binary"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            self.X, self.y = fetch_libsvm(self.dataset)

        data = dict(X=self.X, y=self.y)

        return self.X.shape[1], data

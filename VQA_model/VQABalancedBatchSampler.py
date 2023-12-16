from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, question_type_to_idx):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)
        self.question_type_to_idx = question_type_to_idx
        self.question_type_to_count = self._get_question_type_count()
        self.weights = self._make_weights()

    def _get_question_type_count(self):
        # Count the frequency of each question type
        question_type_count = {qt: 0 for qt in self.question_type_to_idx.values()}
        for _, _, _, question_type_idx, _ in self.dataset.items:
            question_type_count[question_type_idx] += 1
        return question_type_count

    def _make_weights(self):
        # Assign weights inversely proportional to the frequency of the question type
        weights = []
        for _, _, _, question_type_idx, _ in self.dataset.items:
            weight = 1 / self.question_type_to_count[question_type_idx]
            weights.append(weight)
        return weights

    def __iter__(self):
        # Create a sampler with the computed weights
        weighted_sampler = WeightedRandomSampler(self.weights, self.num_samples)
        return iter(weighted_sampler)

    def __len__(self):
        return self.num_samples
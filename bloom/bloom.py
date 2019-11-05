from typing import Any, Generator
import math
import mmh3


class BloomFilter:
    """A Bloom filter.

    A probabilistic data structure[1] used to check if an element is contained
    by a reference set. False negative rate is always guaranteed to be zero,
    and a upper bound for the false positive rate can be approximated for a
    given maximum number of elements.

    The hashing step uses the two-function approach[2], based on two 64 bit
    halves of a 128 bit Murmur3 hash function.

    [1] https://en.wikipedia.org/wiki/Bloom_filter
    [2] Kirsh, Mitzenmacher
        (https://www.eecs.harvard.edu/~michaelm/postscripts/tr-02-05.pdf)

    Attributes:
        n (int): Maximum number of elements in the set.
        fp_rate (float): Maximum desired false positive rate. Only holds if
            'n' is not exceeded.
    """

    def __init__(self, n: int, fp_rate: float) -> None:
        self._check_args(n, fp_rate)
        self.n = n
        self.fp_rate = fp_rate

        # memory size and number of hash functions
        self._calculate_sizes()

        # memory initialization
        self._memory = int(0)

    def add(self, element: Any) -> None:
        """Adds an element to the set, inplace."""
        for idx in self._hash_element(element):
            self._memory |= 1 << idx

    def __contains__(self, element: Any) -> bool:
        """Checks if an element is contained by the set."""
        return all(
            (self._memory | (1 << idx)) == self._memory
            for idx in self._hash_element(element)
        )

    def _check_args(self, n: int, fp_rate: float) -> None:
        """Checks if class args are valid.

        n should be a positive integer.
        fp_rate should be a float in the (0, 1) interval.

        Raises:
            ValueError: Raises if args are incorrect.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("The 'n' parameter should be a positive integer.")

        if not isinstance(fp_rate, float) or not (0 < fp_rate < 1):
            raise ValueError(
                "The 'fp_rate' parameter should be a floating point number "
                "in the (0,1) interval."
            )

    def _calculate_sizes(self) -> None:
        """Calculates number of hash functions and memory size."""
        # k is the number of has functions and partitions
        # k = log2(1/p), where p is the false positive rate
        self._k = int(math.ceil(math.log(1.0 / self.fp_rate, 2)))

        # m is the size of each partition, in bits.
        # m*k is the total memory size.
        # m*k ~= n * log2(1 / p) / ln(2)^2
        self._m = int(
            math.ceil(
                self.n
                * math.log(1 / self.fp_rate, 2)
                * math.log(math.exp(1), 2)
                / self._k
            )
        )

    def _hash_element(self, element: Any) -> Generator[int, None, None]:
        """Hash element using k hash functions, returning indices."""
        if isinstance(element, int):
            element = bin(element)

        h1, h2 = mmh3.hash64(element)
        indices = ((h1 + i * h2) % self._m + i * self._m for i in range(self._k))

        return indices

    def _clear(self) -> None:
        """Clears filter's memory"""
        self._memory = int(0)

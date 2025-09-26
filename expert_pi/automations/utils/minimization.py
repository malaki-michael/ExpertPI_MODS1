import dataclasses
import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
from scipy.optimize import minimize


class MaxNumberIterationsReached(Exception):
    pass


class ValueReached(Exception):
    pass


@dataclasses.dataclass()
class BestResult:
    fn_value: float
    value: np.ndarray | float | None


class MinimizationCallback(Protocol):
    def __call__(self, x: np.ndarray, best_result: BestResult) -> None: ...


class _HandledMinimizedFunction:
    def __init__(
        self,
        minimized_fn: Callable,
        number_of_iterations: int,
        optimum_value: float | None = None,
        callback_fn: MinimizationCallback = lambda x, y: None,
    ) -> None:
        self.number_of_retries = 0
        self.number_of_iterations = number_of_iterations
        self.actual_result = BestResult(fn_value=np.inf, value=None)
        self.minimized_fn = minimized_fn
        self.optimum_value = optimum_value
        self.logger = logging.getLogger(__name__)
        self.callback_fn = callback_fn

    def _store(self, x: np.ndarray, actual_value: float):
        if actual_value < self.actual_result.fn_value:
            self.actual_result = BestResult(fn_value=actual_value, value=x)

    def __call__(self, x: np.ndarray) -> float:
        """This method is a callable which is used to evaluate the function to be minimized at a given point.

        Parameters:
            x (np.ndarray): The point at which the function is to be evaluated.

        Returns:
            float: The value of the function evaluated at the point x.

        Raises:
            MaxNumberIterationsReached: If the maximum number of iterations has been reached.
            ValueReached: If the desired value has been reached.
        """
        self.number_of_retries += 1
        function_value = self.minimized_fn(x)
        self.logger.debug(
            f"ITERATION: {self.number_of_retries}. Actual parameter value: {x}, function value: {function_value}."
        )
        self._store(x, function_value)
        if self.number_of_retries == self.number_of_iterations:
            raise MaxNumberIterationsReached("Max number of iterations reached.")

        if function_value <= self.optimum_value:
            raise ValueReached("Desired value reached.")

        self.callback_fn(x, self.actual_result)
        return function_value

    def reset(self):
        self.actual_result = BestResult(fn_value=np.inf, value=None)
        self.number_of_retries = 0


class FunctionMinimizer:
    def __init__(
        self,
        minimized_fn: Callable,
        optimum_value: float = -np.inf,
        callback_fn: MinimizationCallback = lambda x, y: None,
    ) -> None:
        self.minimized_fn = minimized_fn
        self.optimum_value = optimum_value
        self.callback_fn = callback_fn
        self.logger = logging.getLogger(__name__)
        self.result: BestResult | None = None

    def execute(
        self, initial_guess: np.ndarray, number_of_iterations: int, method: str = "Nelder-Mead", options=None
    ) -> np.ndarray | float | None:
        """Executes the minimization process for the function to be minimized.

        This method uses the scipy.optimize.minimize method to find the minimum of a function.
        The method used for minimization can be specified by the 'method' parameter.
        The 'options' parameter can be used to specify options that are passed to the minimization method.

        Parameters:
            initial_guess (np.ndarray): The initial guess for the minimum.
            number_of_iterations (int): The maximum number of evaluations for the minimization process.
            method (str, optional): The method to be used for minimization. Defaults to "Nelder-Mead".
            options (dict, optional): Options to be passed to the minimization method. Defaults to None.

        Returns:
            np.ndarray | float | None: The value of the function at the minimum.

        Raises:
            MaxNumberIterationsReached: If the maximum number of iterations has been reached.
            ValueReached: If the desired value has been reached.
        """
        if options is None:
            options = {"disp": True}

        minimized_fn = _HandledMinimizedFunction(
            minimized_fn=self.minimized_fn,
            number_of_iterations=number_of_iterations,
            optimum_value=self.optimum_value,
            callback_fn=self.callback_fn,
        )

        try:
            minimize(minimized_fn, initial_guess, method=method, options=options)
        except (MaxNumberIterationsReached, ValueReached) as e:
            self.logger.info(e)
        except Exception as e:
            # NOTE: This is a catch-all exception handler.
            # It is needed for user callbacks which can raise any exception.
            self.logger.error(e)
        finally:
            self.result = minimized_fn.actual_result
            output_value = minimized_fn.actual_result.value
            self.logger.debug(output_value)

        self.logger.info(f"Optimized value for function {self.minimized_fn.__name__}: {output_value}")
        return output_value

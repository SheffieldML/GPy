# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import inspect
import pkgutil
import os


def check_grad(Model):
    assert Model.checkgrad(), "Gradient check failed!"


def check_model_instance(Model):
    assert isinstance(Model, GPy.models), "Wrong type!"


def model_checkgrads(model):
    model.randomize()
    # NOTE: Step as 1e-4, this should be acceptable for more peaky models
    return model.checkgrad(step=1e-4)


def model_instance(model):
    return isinstance(model, GPy.core.model.Model)


def flatten_nested(lst):
    result = []
    for element in lst:
        if hasattr(element, "__iter__"):
            result.extend(flatten_nested(element))
        else:
            result.append(element)
    return result


def test_models():
    # TODO: testing setup is not that clear to me yet...
    optimize = False
    plot = True
    examples_path = os.path.dirname(GPy.examples.__file__)
    # Load modules
    failing_models = {}
    for loader, module_name, _is_pkg in pkgutil.iter_modules([examples_path]):
        # Load examples
        module_examples = loader.find_module(module_name).load_module(module_name)
        print("MODULE", module_examples)
        print("Before")
        print(inspect.getmembers(module_examples, predicate=inspect.isfunction))
        functions = [
            func
            for func in inspect.getmembers(
                module_examples, predicate=inspect.isfunction
            )
            if func[0].startswith("_") is False
        ][::-1]
        print("After")
        print(functions)
        for example in functions:
            if example[0] in ["epomeo_gpx"]:
                # These are the edge cases that we might want to handle specially
                if example[0] == "epomeo_gpx" and not GPy.util.datasets.gpxpy_available:
                    print("Skipping as gpxpy is not available to parse GPS")
                    continue

            print("Testing example: ", example[0])
            # Generate model

            try:
                models = [example[1](optimize=optimize, plot=plot)]
                # If more than one model returned, flatten them
                models = flatten_nested(models)
            except Exception as e:
                failing_models[example[0]] = "Cannot make model: \n{e}".format(e=e)
            else:
                print(models)
                model_checkgrads.description = "test_checkgrads_%s" % example[0]
                try:
                    for model in models:
                        if not model_checkgrads(model):
                            failing_models[model_checkgrads.description] = False
                except Exception as e:
                    failing_models[model_checkgrads.description] = e

                model_instance.description = "test_instance_%s" % example[0]
                try:
                    for model in models:
                        if not model_instance(model):
                            failing_models[model_instance.description] = False
                except Exception as e:
                    failing_models[model_instance.description] = e

            # yield model_checkgrads, model
            # yield model_instance, model

        print("Finished checking module {m}".format(m=module_name))
        if len(failing_models.keys()) > 0:
            print("Failing models: ")
            print(failing_models)

    if len(failing_models.keys()) > 0:
        print(failing_models)
        raise Exception(failing_models)

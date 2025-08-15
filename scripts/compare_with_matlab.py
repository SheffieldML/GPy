#!/usr/bin/env python3
"""
Comparison script for LFM kernel implementation against MATLAB reference.

This script generates test cases and compares results between GPy and MATLAB
implementations to validate our LFM kernel implementation.
"""

import numpy as np
import subprocess
import tempfile
import os
import json
from pathlib import Path


class MATLABComparison:
    """Compare GPy LFM kernel results with MATLAB reference."""
    
    def __init__(self, matlab_path=None):
        """Initialize with path to MATLAB/Octave executable."""
        self.matlab_path = matlab_path or self._find_matlab()
        self.temp_dir = tempfile.mkdtemp()
        
    def _find_matlab(self):
        """Try to find MATLAB or Octave executable."""
        # Try common MATLAB paths
        matlab_paths = [
            '/Applications/MATLAB_R2023b.app/bin/matlab',  # macOS
            '/usr/local/bin/matlab',  # Linux
            'matlab',  # In PATH
        ]
        
        # Try Octave as fallback
        octave_paths = [
            '/usr/local/bin/octave',  # macOS
            '/usr/bin/octave',  # Linux
            'octave',  # In PATH
        ]
        
        for path in matlab_paths + octave_paths:
            try:
                result = subprocess.run([path, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"Found MATLAB/Octave: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        raise RuntimeError("Could not find MATLAB or Octave executable")
    
    def create_matlab_script(self, test_case):
        """Create MATLAB script for kernel computation."""
        script = f"""
% MATLAB script for LFM kernel comparison
% Generated test case: {test_case['name']}

% Add GPmat to path (assuming it's in ~/lawrennd/GPmat)
addpath('~/lawrennd/GPmat/matlab');

% Test parameters
{self._matlab_params(test_case)}

% Create kernel
{self._matlab_kernel_creation(test_case)}

% Compute kernel matrix
{self._matlab_kernel_computation(test_case)}

% Save results
results = struct();
results.K = K;
results.params = {self._matlab_params_struct(test_case)};
results.test_case = '{test_case['name']}';

save('{os.path.join(self.temp_dir, 'matlab_results.mat')}', 'results');
fprintf('MATLAB computation completed\\n');
"""
        return script
    
    def _matlab_params(self, test_case):
        """Generate MATLAB parameter definitions."""
        params = test_case.get('params', {})
        lines = []
        for key, value in params.items():
            if isinstance(value, (int, float)):
                lines.append(f"{key} = {value};")
            elif isinstance(value, list):
                lines.append(f"{key} = [{', '.join(map(str, value))}];")
        return '\n'.join(lines)
    
    def _matlab_kernel_creation(self, test_case):
        """Generate MATLAB kernel creation code."""
        kernel_type = test_case.get('kernel_type', 'sim')
        if kernel_type == 'sim':
            return """
% Create SIM kernel
kern = kernCreate(t, 'sim');
kern.decay = decay;
kern.delay = delay;
kern.variance = variance;
kern.inverseWidth = inverseWidth;
kern = kernParamInit(kern);
"""
        elif kernel_type == 'disim':
            return """
% Create DISIM kernel
kern = kernCreate(t, 'disim');
kern.decay = decay;
kern.di_decay = di_decay;
kern.variance = variance;
kern.di_variance = di_variance;
kern.inverseWidth = inverseWidth;
kern.rbf_variance = rbf_variance;
kern = kernParamInit(kern);
"""
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def _matlab_kernel_computation(self, test_case):
        """Generate MATLAB kernel computation code."""
        return """
% Compute kernel matrix
K = kernCompute(kern, t);
"""
    
    def _matlab_params_struct(self, test_case):
        """Generate MATLAB parameter structure."""
        params = test_case.get('params', {})
        param_str = ', '.join([f"'{k}', {v}" for k, v in params.items()])
        return f"struct({param_str})"
    
    def run_matlab_computation(self, test_case):
        """Run MATLAB computation for given test case."""
        script = self.create_matlab_script(test_case)
        
        # Write script to temporary file
        script_path = os.path.join(self.temp_dir, 'compute_kernel.m')
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Run MATLAB
        try:
            if 'octave' in self.matlab_path.lower():
                cmd = [self.matlab_path, '--no-gui', '--silent', script_path]
            else:
                cmd = [self.matlab_path, '-batch', f"run('{script_path}')"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"MATLAB error: {result.stderr}")
                return None
                
            # Load results
            import scipy.io
            results_path = os.path.join(self.temp_dir, 'matlab_results.mat')
            if os.path.exists(results_path):
                matlab_data = scipy.io.loadmat(results_path)
                return matlab_data['results'][0, 0]
            else:
                print("MATLAB results file not found")
                return None
                
        except subprocess.TimeoutExpired:
            print("MATLAB computation timed out")
            return None
        except Exception as e:
            print(f"Error running MATLAB: {e}")
            return None
    
    def create_test_cases(self):
        """Create standard test cases for comparison."""
        test_cases = [
            {
                'name': 'sim_basic',
                'kernel_type': 'sim',
                'params': {
                    'decay': 1.0,
                    'delay': 0.1,
                    'variance': 1.0,
                    'inverseWidth': 1.0
                },
                't': np.linspace(0, 5, 20).reshape(-1, 1)
            },
            {
                'name': 'sim_fast_decay',
                'kernel_type': 'sim',
                'params': {
                    'decay': 2.0,
                    'delay': 0.0,
                    'variance': 1.0,
                    'inverseWidth': 0.5
                },
                't': np.linspace(0, 3, 15).reshape(-1, 1)
            },
            {
                'name': 'disim_basic',
                'kernel_type': 'disim',
                'params': {
                    'decay': 1.0,
                    'di_decay': 0.5,
                    'variance': 1.0,
                    'di_variance': 1.0,
                    'inverseWidth': 1.0,
                    'rbf_variance': 1.0
                },
                't': np.linspace(0, 5, 20).reshape(-1, 1)
            }
        ]
        return test_cases
    
    def compare_results(self, matlab_results, gpy_results, test_case):
        """Compare MATLAB and GPy results."""
        if matlab_results is None or gpy_results is None:
            return {'status': 'error', 'message': 'Missing results'}
        
        # Extract kernel matrices
        matlab_K = matlab_results['K']
        gpy_K = gpy_results['K']
        
        # Basic shape check
        if matlab_K.shape != gpy_K.shape:
            return {
                'status': 'error',
                'message': f'Shape mismatch: MATLAB {matlab_K.shape} vs GPy {gpy_K.shape}'
            }
        
        # Compute differences
        abs_diff = np.abs(matlab_K - gpy_K)
        rel_diff = abs_diff / (np.abs(matlab_K) + 1e-10)
        
        comparison = {
            'status': 'success',
            'test_case': test_case['name'],
            'shapes_match': True,
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_abs_diff': float(np.mean(abs_diff)),
            'max_rel_diff': float(np.max(rel_diff)),
            'mean_rel_diff': float(np.mean(rel_diff)),
            'matlab_shape': matlab_K.shape,
            'gpy_shape': gpy_K.shape
        }
        
        # Check if results are close enough
        tolerance = 1e-6
        comparison['within_tolerance'] = comparison['max_abs_diff'] < tolerance
        
        return comparison
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main comparison function."""
    print("LFM Kernel MATLAB Comparison")
    print("=" * 40)
    
    # Initialize comparison framework
    try:
        comparator = MATLABComparison()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please ensure MATLAB or Octave is installed and in PATH")
        return
    
    # Create test cases
    test_cases = comparator.create_test_cases()
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 20)
        
        # Run MATLAB computation
        print("Running MATLAB computation...")
        matlab_results = comparator.run_matlab_computation(test_case)
        
        if matlab_results is None:
            print("MATLAB computation failed")
            continue
        
        # TODO: Run GPy computation (when implemented)
        print("GPy computation not yet implemented")
        gpy_results = None
        
        # Compare results
        if gpy_results is not None:
            comparison = comparator.compare_results(matlab_results, gpy_results, test_case)
            results.append(comparison)
            
            if comparison['status'] == 'success':
                print(f"✓ Shapes match: {comparison['shapes_match']}")
                print(f"✓ Max abs diff: {comparison['max_abs_diff']:.2e}")
                print(f"✓ Within tolerance: {comparison['within_tolerance']}")
            else:
                print(f"✗ Error: {comparison['message']}")
        else:
            print("Skipping comparison (GPy not implemented yet)")
    
    # Save results
    results_file = 'matlab_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Cleanup
    comparator.cleanup()


if __name__ == "__main__":
    main()

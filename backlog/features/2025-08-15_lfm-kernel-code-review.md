---
id: "lfm-kernel-code-review"
title: "Review existing LFM kernel implementations"
status: "Completed"
priority: "High"
created: "2025-08-15"
last_updated: "2025-08-15"
owner: "Neil Lawrence"
github_issue: ""
dependencies: ""
tags:
- lfm
- kernel
- code-review
- documentation
---

# Review existing LFM kernel implementations

## Description
Conduct a comprehensive review of existing LFM (Latent Force Model) kernel implementations in both GPy and MATLAB to understand the current state, design decisions, and limitations.

## Background
- GPy has existing ODE-based kernels (`EQ_ODE1`, `EQ_ODE2`) that implement LFM concepts
- MATLAB implementation in GPmat provides a more complete LFM framework
- Need to understand differences and identify modernization opportunities

## Tasks
- [x] Review `GPy/kern/src/eq_ode1.py` and `eq_ode2.py` implementations
- [x] Analyze MATLAB LFM implementation structure and patterns
- [x] Document current limitations and inconsistencies
- [ ] Identify reusable components and design patterns
- [ ] Compare parameter handling approaches
- [ ] Review cross-kernel computation methods
- [ ] Document mathematical foundations and implementation details

## Acceptance Criteria
- [ ] Complete documentation of existing implementations
- [ ] Clear understanding of design differences between GPy and MATLAB versions
- [ ] Identified list of modernization opportunities
- [ ] Documentation of mathematical foundations
- [ ] Assessment of current limitations and bugs

## Implementation Notes
- Focus on understanding the mathematical foundations from the papers
- Pay attention to parameter tying and multi-output handling
- Document the differential equation structure and kernel computation
- Identify opportunities for using GPy's modern multioutput kernel approach

## Related
- CIP: 0001 (LFM kernel implementation)
- Papers: Álvarez et al. (2009, 2012), Lawrence et al. (2006)
- Backlog: parameter-tying-framework (fundamental dependency)

## Progress Updates

### 2025-08-15
Started code review task. Initial findings:

**GPy Implementations:**
- `EQ_ODE1`: First-order differential equation kernel with decay rates and sensitivities
- `EQ_ODE2`: Second-order differential equation kernel with spring/damper constants
- Both use GPy's multioutput approach with output index as second input dimension
- Complex kernel computation with multiple covariance types (Kuu, Kfu, Kuf, Kusu)
- Uses `@Cache_this` decorator for performance optimization

**GPmat Implementation:**
- More complete framework with `lfmCreate`, `lfmKernCompute`, `lfmKernParamInit`
- Uses multi-kernel approach with parameter tying
- Supports multiple displacements driven by multiple forces
- Cleaner separation of concerns with dedicated model creation

**Key Differences:**
- GPy uses single kernel class per ODE order, GPmat uses multi-kernel composition
- GPy has more complex index handling for multioutput
- GPmat has better parameter organization and tying mechanisms
- **Critical Gap**: GPy lacks parameter tying framework (GPmat has `modelTieParam()`)

### 2025-08-15 (Parameter Tying Discovery)
**Major Finding**: Identified parameter tying as a fundamental limitation affecting LFM implementation:
- Created backlog item for parameter tying investigation
- Found 5+ years of GitHub issues requesting this functionality
- Related to paramz framework limitation (documented but not implemented)
- Created CIP-0002 for community discussion of parameter tying solutions
- **Decision**: Proceed with LFM implementation assuming parameter tying will be addressed separately
- **Rationale**: Keeps implementation clean and focused on core LFM functionality

### 2025-08-15 (MATLAB Kernel Analysis)
**Comprehensive MATLAB Analysis**: Examined complete kernel implementations in GPmat:

**SIM Kernel (First-order ODE):**
- Parameters: `delay`, `decay`, `initVal`, `variance`, `inverseWidth`
- Differential equation: `dx(t)/dt = B + S f(t-delta) - D x(t)`
- Uses `simComputeH()` for kernel computation with error functions
- Supports Gaussian initial conditions and negative sensitivity options
- Cross-kernel computation with RBF kernels via `simXrbfKernCompute()`

**DISIM Kernel (Second-order ODE):**
- Parameters: `di_decay`, `inverseWidth`, `di_variance`, `decay`, `variance`, `rbf_variance`
- Two-level differential equation system
- More complex parameter structure for hierarchical modeling
- Cross-kernel computations with SIM, RBF, and other DISIM kernels

**Key Insights:**
- SIM/DISIM are specialized kernels for gene networks
- LFM is the general framework that can use these kernels
- Complex cross-kernel computation system for multi-output modeling
- Error function-based computation (`lnDiffErfs`) for analytical solutions
- Parameter constraints and transformations built into kernel structure
import arviz as az
import sys

# Configuration
trace_path = "outputs/dynamic_gp_two_timescale_pen100/trace.zarr"

print(f"--- Inspecting variables in posterior group of: {trace_path} ---")

try:
    # Load inference data using from_zarr
    idata = az.from_zarr(trace_path)
    
    if "posterior" in idata:
        posterior_vars = list(idata.posterior.data_vars)
        print("\nVariables found in posterior group:")
        for var in sorted(posterior_vars):
            print(f"- {var}")
    else:
        print("\nError: No 'posterior' group found in the InferenceData object.")

except FileNotFoundError:
    print(f"\nError: Trace file not found at {trace_path}")
    sys.exit(1)
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n--- Inspection Complete ---") 
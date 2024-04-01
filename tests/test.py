from EMOGEA.prepare_data import prepare_data
from EMOGEA.load_sample import load_sample
from EMOGEA.ml_projection import ml_projection
from EMOGEA.multivariate_curve_resolution import multivariate_curve_resolution

if __name__ == "__main__":
    data, meta_data = load_sample()
    X, Xres, Xcov = prepare_data(data, meta_data, apply_log_transformation=False)
    #U,S,V,Xest1 = ml_projection(
    #    X, Xcov
    #)
    P,C,Xest2,gene_profiles = multivariate_curve_resolution(
        X, Xres, number_of_components=3,
        init_algorithm="simplisma"
    )
    print("Doen")
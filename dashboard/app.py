import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


MODEL_PATH = "models/gradient_boosting_final.joblib"
DATA_PATH = "data/processed/train_features_no_outliers.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def main():
    data = load_data()
    st.set_page_config(
        page_title="Housing Price Prediction",
        page_icon="🏠",
        layout="wide",
    )

    st.title("🏠 Housing Price Prediction")
    st.write("Estimate the sale price of a house using its main characteristics.")

    model = load_model()
    df = load_data()

    with st.form("house_form"):

        st.subheader("🏠 Main characteristics")

        col1, col2, col3 = st.columns(3)

        with col1:
            overall_qual = st.slider(
                "Overall Quality",
                1,
                10,
                5,
            )

        with col2:
            gr_liv_area = st.number_input(
                "Living Area (sq ft)",
                300,
                6000,
                1500,
            )

        with col3:
            year_built = st.number_input(
                "Year Built",
                1800,
                2026,
                2000,
            )

        st.subheader("📐 Surface and rooms")

        col1, col2, col3 = st.columns(3)

        with col1:
            total_bsmt_sf = st.number_input(
                "Basement Area (sq ft)",
                0,
                4000,
                1000,
            )

        with col2:
            first_flr_sf = st.number_input(
                "1st Floor Area (sq ft)",
                300,
                4000,
                1000,
            )

        with col3:
            second_flr_sf = st.number_input(
                "2nd Floor Area (sq ft)",
                0,
                3000,
                500,
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            tot_rms_abv_grd = st.number_input(
                "Rooms Above Ground",
                2,
                20,
                6,
            )

        with col2:
            full_bath = st.number_input(
                "Full Bathrooms",
                0,
                5,
                2,
            )

        with col3:
            lot_area = st.number_input(
                "Lot Area (sq ft)",
                500,
                200000,
                8000,
            )

        st.subheader("🚗 Garage")

        col1, col2, col3 = st.columns(3)

        with col1:
            garage_cars = st.number_input(
                "Garage Cars",
                0,
                5,
                2,
            )

        with col2:
            garage_area = st.number_input(
                "Garage Area (sq ft)",
                0,
                1500,
                500,
            )

        with col3:
            garage_yr_blt = st.number_input(
                "Garage Year Built",
                1800,
                2026,
                2000,
            )

        st.subheader("🏡 Quality")

        col1, col2, col3 = st.columns(3)

        with col1:
            kitchen_qual = st.selectbox(
                "Kitchen Quality",
                sorted(df["KitchenQual"].dropna().unique()),
            )

        with col2:
            exter_qual = st.selectbox(
                "Exterior Quality",
                sorted(df["ExterQual"].dropna().unique()),
            )

        with col3:
            bsmt_qual = st.selectbox(
                "Basement Quality",
                sorted(df["BsmtQual"].dropna().unique()),
            )

        st.subheader("🏘️ House type")

        col1, col2, col3 = st.columns(3)

        with col1:
            garage_finish = st.selectbox(
                "Garage Finish",
                sorted(df["GarageFinish"].dropna().unique()),
            )

        with col2:
            bsmt_exposure = st.selectbox(
                "Basement Exposure",
                sorted(df["BsmtExposure"].dropna().unique()),
            )

        with col3:
            house_style = st.selectbox(
                "House Style",
                sorted(df["HouseStyle"].dropna().unique()),
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            garage_type = st.selectbox(
                "Garage Type",
                sorted(df["GarageType"].dropna().unique()),
            )

        with col2:
            exterior_1st = st.selectbox(
                "Exterior",
                sorted(df["Exterior1st"].dropna().unique()),
            )

        with col3:
            sale_condition = st.selectbox(
                "Sale Condition",
                sorted(df["SaleCondition"].dropna().unique()),
            )

        st.subheader("📅 Renovation and sale")

        col1, col2, col3 = st.columns(3)

        with col1:
            year_remod_add = st.number_input(
                "Year Remodeled",
                1800,
                2026,
                2005,
            )

        with col2:
            yr_sold = st.number_input(
                "Year Sold",
                2006,
                2026,
                2010,
            )

        with col3:
            fireplaces = st.number_input(
                "Fireplaces",
                0,
                5,
                1,
            )

        predict = st.form_submit_button(
            "💰 Predict House Price"
        )

    if predict:

        input_data = pd.DataFrame([{
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod_add,
            "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual,
            "BsmtQual": bsmt_qual,
            "GarageFinish": garage_finish,
            "2ndFlrSF": second_flr_sf,
            "FullBath": full_bath,
            "TotRmsAbvGrd": tot_rms_abv_grd,
            "GarageYrBlt": garage_yr_blt,
            "Fireplaces": fireplaces,
            "MasVnrArea": 0,
            "BsmtFinSF1": total_bsmt_sf * 0.5,
            "LotArea": lot_area,
            "BsmtExposure": bsmt_exposure,
            "GarageType": garage_type,
            "HouseStyle": house_style,
            "Exterior1st": exterior_1st,
            "SaleCondition": sale_condition,
            "YrSold": yr_sold,
            "HalfBath": 0,
            "BsmtFullBath": 0,
            "BsmtHalfBath": 0,
        }])

        input_data["TotalSF"] = (
            input_data["TotalBsmtSF"]
            + input_data["1stFlrSF"]
            + input_data["2ndFlrSF"]
        )

        input_data["TotalBathrooms"] = (
            input_data["FullBath"]
            + 0.5 * input_data["HalfBath"]
            + input_data["BsmtFullBath"]
            + 0.5 * input_data["BsmtHalfBath"]
        )

        input_data["HouseAge"] = (
            input_data["YrSold"]
            - input_data["YearBuilt"]
        )

        input_data["YearsSinceRemod"] = (
            input_data["YrSold"]
            - input_data["YearRemodAdd"]
        ).clip(lower=0)

        input_data = input_data.drop(
            columns=[
                "YrSold",
                "HalfBath",
                "BsmtFullBath",
                "BsmtHalfBath",
            ]
        )

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")

        st.metric(
            "Estimated Sale Price",
            f"${prediction:,.0f}",
        )

        st.info(
            f"Estimated price: ${prediction:,.0f}"
        )
        st.subheader("Distribution des prix")

        fig, ax = plt.subplots()
  
        ax.hist(data["SalePrice"], bins=30)
        ax.axvline(prediction, linestyle="--", linewidth=2)

        ax.set_xlabel("Prix")
        ax.set_ylabel("Nombre de logements")

        st.pyplot(fig)


if __name__ == "__main__":
    main()
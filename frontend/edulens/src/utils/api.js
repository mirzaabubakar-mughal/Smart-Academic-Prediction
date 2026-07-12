export const API_URL = 'http://localhost:8000';

// API endpoints
export const ENDPOINTS = {
  // Dataset endpoints
  DATASET_PREVIEW: `${API_URL}/dataset/preview`,
  DATASET_INFO: `${API_URL}/dataset/info`,
  DATASET_SMOTE: `${API_URL}/dataset/smote_preview`,

  // Training endpoints
  TRAIN: `${API_URL}/train`,

  // Prediction endpoints
  PREDICT: `${API_URL}/predict`,

  // Visualization endpoints
  VIZ_2D: `${API_URL}/visualization/2d_marks_vs_grade`,
  VIZ_3D: `${API_URL}/visualization/3d_feature_space`,
  VIZ_PCA: `${API_URL}/visualization/pca`,
  VIZ_CLASS_DIST: `${API_URL}/visualization/class_distribution`,
  VIZ_MODEL_COMPARE: `${API_URL}/visualization/model_comparison`,
  VIZ_FEATURE_CORR: `${API_URL}/visualization/feature_correlation`,
  VIZ_BEST_MATRIX: `${API_URL}/visualization/best_model_matrix`,
  VIZ_BEST_ROC: `${API_URL}/visualization/best_model_roc`,

  // Database endpoints
  DB_STATUS: `${API_URL}/database/status`,
  DB_TRAINING_HISTORY: `${API_URL}/database/training_history`,
  DB_PREDICTION_HISTORY: `${API_URL}/database/prediction_history`,
  DB_MODEL_COMPARISON: `${API_URL}/database/model_comparison`,
};

// Generic API call function
export const apiCall = async (url, options = {}) => {
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'API request failed');
    }

    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};
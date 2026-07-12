import React, { useState } from 'react';
import { 
  FaRocket, 
  FaSync, 
  FaExclamationTriangle, 
  FaCheckCircle,
  FaChartLine,
  FaGraduationCap,
  FaClock
} from 'react-icons/fa';
import { ENDPOINTS, apiCall } from '../utils/api';

function Prediction({
  showMessage,
  setPredictions,
  predictions,
  modelsTrained,
  loadPredictionHistory,
}) {
  // ============ STATE ============
  const [form, setForm] = useState({
    midterm_marks: 15,
    sessional_marks: 15,
    final_exam_marks: 35,
    class_participation: 'Medium',
    study_hours_perday: 5,
    sleep_hours_perday: 7,
    previous_cgpa: 3.0,
  });
  const [loading, setLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  // ============ CONSTANTS ============
  const modelDisplayNames = {
    'knn_euclidean': 'KNN (Euclidean)',
    'knn_manhattan': 'KNN (Manhattan)',
    'multinomial_nb': 'Multinomial Naive Bayes',
    'svm_rbf': 'Non-Linear SVM (RBF)',
    'neuralnetwork_softmax': 'Neural Network (Softmax)',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
  };

  // ============ HELPERS ============
  const getGradeColor = (grade) => {
    const colors = { A: '#3fb950', B: '#58a6ff', C: '#d29922', D: '#f0883e', F: '#f85149' };
    return colors[grade] || '#8b949e';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 85) return '#3fb950';
    if (confidence >= 70) return '#d29922';
    return '#f85149';
  };

  const getConfidenceText = (confidence) => {
    if (confidence >= 85) return 'High';
    if (confidence >= 70) return 'Medium';
    return 'Low';
  };

  // ============ HANDLERS ============
  const handlePrediction = async (e) => {
    e.preventDefault();
    if (!modelsTrained) {
      showMessage('error', 'Please train the models first!');
      setPredictionError('Models not trained. Please go to Training tab first.');
      return;
    }

    setLoading(true);
    setPredictionError(null);

    try {
      const result = await apiCall(ENDPOINTS.PREDICT, {
        method: 'POST',
        body: JSON.stringify({
          midterm_marks: parseFloat(form.midterm_marks),
          sessional_marks: parseFloat(form.sessional_marks),
          final_exam_marks: parseFloat(form.final_exam_marks),
          class_participation: form.class_participation,
          study_hours_perday: parseFloat(form.study_hours_perday),
          sleep_hours_perday: parseFloat(form.sleep_hours_perday),
          previous_cgpa: parseFloat(form.previous_cgpa),
        }),
      });

      setPredictions(result);
      if (loadPredictionHistory) loadPredictionHistory();
      showMessage('success', `Prediction successful! Grade: ${result.final_prediction}`);
    } catch (err) {
      setPredictionError(err.message);
      showMessage('error', err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm({
      midterm_marks: null,
      sessional_marks: null,
      final_exam_marks: null,
      class_participation: null,
      study_hours_perday: null,
      sleep_hours_perday: null,
      previous_cgpa: null,
    });
    showMessage('info', 'Form reset');
  };

  // ============ RENDER ============
  return (
    <div className="fade-in">
      {/* Prediction Form */}
      <div className="card">
        <div className="card-title">
          <FaGraduationCap />
          <span>Predict Student Grade</span>
        </div>

        {!modelsTrained && (
          <div className="alert alert-error">
            <FaExclamationTriangle /> Models not trained yet. Please go to Training tab first.
          </div>
        )}

        {predictionError && (
          <div className="alert alert-error">
            <FaExclamationTriangle /> {predictionError}
          </div>
        )}

        <form onSubmit={handlePrediction}>
          <div className="form-grid">
            <div className="form-group">
              <label>Midterm Marks (0-25)</label>
              <input
                type="number"
                min="0"
                max="25"
                step="0.5"
                value={form.midterm_marks}
                onChange={(e) => setForm({ ...form, midterm_marks: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Sessional Marks (0-25)</label>
              <input
                type="number"
                min="0"
                max="25"
                step="0.5"
                value={form.sessional_marks}
                onChange={(e) => setForm({ ...form, sessional_marks: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Final Exam Marks (0-50)</label>
              <input
                type="number"
                min="0"
                max="50"
                step="0.5"
                value={form.final_exam_marks}
                onChange={(e) => setForm({ ...form, final_exam_marks: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Study Hours/Day (0-12)</label>
              <input
                type="number"
                min="0"
                max="12"
                step="0.5"
                value={form.study_hours_perday}
                onChange={(e) => setForm({ ...form, study_hours_perday: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Sleep Hours/Day (4-10)</label>
              <input
                type="number"
                min="4"
                max="10"
                step="0.5"
                value={form.sleep_hours_perday}
                onChange={(e) => setForm({ ...form, sleep_hours_perday: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Previous CGPA (0-4)</label>
              <input
                type="number"
                min="0"
                max="4"
                step="0.1"
                value={form.previous_cgpa}
                onChange={(e) => setForm({ ...form, previous_cgpa: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label>Class Participation</label>
              <select
                value={form.class_participation}
                onChange={(e) => setForm({ ...form, class_participation: e.target.value })}
              >
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
              </select>
            </div>
          </div>

          <div className="form-actions">
            <button type="submit" className="btn-primary" disabled={loading || !modelsTrained}>
              {loading ? <><FaSync className="spinner" /> Predicting...</> : <><FaRocket /> Predict Grade</>}
            </button>
            <button type="button" className="btn-secondary" onClick={resetForm}>
              <FaSync /> Reset Form
            </button>
          </div>
        </form>
      </div>

      {/* Prediction Results */}
      {predictions && predictions.predictions && (
        <div className="card fade-in">
          <div className="card-title">
            <FaChartLine />
            <span>Prediction Results</span>
          </div>

          {/* Overall Confidence */}
          {predictions.overall_confidence && (
            <div
              className={`confidence-banner ${getConfidenceText(predictions.overall_confidence).toLowerCase()}`}
              style={{
                marginBottom: '20px',
                padding: '16px',
                borderRadius: '12px',
                border: `1px solid ${getConfidenceColor(predictions.overall_confidence)}`,
                background: `rgba(${predictions.overall_confidence >= 85 ? '63, 185, 80' : predictions.overall_confidence >= 70 ? '210, 153, 34' : '248, 81, 73'}, 0.05)`,
              }}
            >
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px' }}>
                <div>
                  <div style={{ fontSize: '14px', color: '#8b949e' }}>Overall Prediction Confidence</div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: getConfidenceColor(predictions.overall_confidence) }}>
                    {predictions.overall_confidence}%
                  </div>
                </div>
                <div style={{ flex: 1, minWidth: '200px' }}>
                  <div style={{ width: '100%', height: '8px', background: '#161b22', borderRadius: '4px', overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${predictions.overall_confidence}%`,
                        height: '100%',
                        background: getConfidenceColor(predictions.overall_confidence),
                        transition: 'width 0.5s ease',
                      }}
                    />
                  </div>
                  <div style={{ marginTop: '8px', fontSize: '12px', color: '#8b949e' }}>
                    {predictions.overall_confidence >= 85
                      ? '✅ High confidence - Prediction is reliable'
                      : predictions.overall_confidence >= 70
                      ? '⚠️ Medium confidence - Consider additional factors'
                      : '❌ Low confidence - Borderline case, review manually'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Grade Display */}
          <div
            className="grade-display"
            style={{
              padding: '24px',
              borderRadius: '12px',
              border: `2px solid ${getGradeColor(predictions.final_prediction)}`,
              textAlign: 'center',
              marginBottom: '20px',
            }}
          >
            <div style={{ fontSize: '56px', fontWeight: '800', color: getGradeColor(predictions.final_prediction) }}>
              {predictions.final_prediction}
            </div>
            <div style={{ fontSize: '18px', fontWeight: '600' }}>{predictions.remarks?.short || ''}</div>
            <div style={{ marginTop: '12px', color: '#8b949e' }}>Score: {(predictions.total_marks || 0).toFixed(1)}/100</div>
            <div style={{ width: '200px', height: '8px', background: '#161b22', borderRadius: '4px', margin: '8px auto', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${(predictions.percentage || 0)}%`,
                  height: '100%',
                  background: getGradeColor(predictions.final_prediction),
                }}
              />
            </div>

            {/* Remarks */}
            <div style={{ marginTop: '16px', padding: '16px', background: 'var(--bg-secondary)', borderRadius: '12px', textAlign: 'left' }}>
              <div style={{ fontWeight: '600', marginBottom: '8px' }}>📝 Remarks</div>
              <div style={{ fontSize: '14px', color: '#8b949e' }}>{predictions.remarks?.detailed || ''}</div>
              <div style={{ marginTop: '12px', padding: '12px', background: 'var(--bg-card)', borderRadius: '8px' }}>
                <div style={{ fontWeight: '600', marginBottom: '8px' }}>💡 Improvement Suggestions</div>
                <div style={{ fontSize: '13px', color: '#8b949e' }}>{predictions.remarks?.improvement || ''}</div>
              </div>
              {predictions.remarks?.confidence_note && (
                <div style={{ marginTop: '12px', padding: '12px', background: 'rgba(88, 166, 255, 0.1)', borderRadius: '8px', color: '#58a6ff', fontSize: '13px' }}>
                  {predictions.remarks.confidence_note}
                </div>
              )}
            </div>
          </div>

          {/* All Model Predictions */}
          <h4 style={{ marginBottom: '16px', fontSize: '16px', fontWeight: '600' }}>All Model Predictions with Confidence</h4>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Predicted Grade</th>
                  <th>Confidence</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(predictions.predictions).map(([model, grade]) => {
                  const confidence = predictions.confidences ? predictions.confidences[model] : 85;
                  const confidenceColor = getConfidenceColor(confidence);
                  const confidenceText = getConfidenceText(confidence);
                  const displayName = modelDisplayNames[model] || model;

                  return (
                    <tr key={model}>
                      <td className="model-name">{displayName}</td>
                      <td>
                        <span className={`badge badge-${grade.toLowerCase()}`}>{grade}</span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ width: '80px', height: '6px', background: '#161b22', borderRadius: '4px', overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${confidence}%`,
                                height: '100%',
                                background: confidenceColor,
                                borderRadius: '4px',
                                transition: 'width 0.3s ease',
                              }}
                            />
                          </div>
                          <span style={{ fontSize: '12px', fontWeight: '600', color: confidenceColor, minWidth: '45px' }}>
                            {confidence}%
                          </span>
                        </div>
                      </td>
                      <td>
                        <span
                          className="status-badge"
                          style={{
                            padding: '4px 8px',
                            borderRadius: '12px',
                            fontSize: '11px',
                            fontWeight: '600',
                            background: `${confidenceColor}20`,
                            color: confidenceColor,
                          }}
                        >
                          {confidenceText} Confidence
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default Prediction;
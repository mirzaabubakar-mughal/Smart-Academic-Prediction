import React, { useState, useEffect } from 'react';
import { 
  FaHistory, 
  FaDatabase, 
  FaTable, 
  FaChartBar, 
  FaAward, 
  FaClock, 
  FaExclamationTriangle,
  FaCheckCircle,
  FaSpinner,
  FaFilter
} from 'react-icons/fa';

function History({
  showMessage,
  trainingHistory,
  predictionHistory,
  modelComparison,
  loadTrainingHistory,
  loadPredictionHistory,
  loadModelComparison,
  dbConnected,
}) {
  // ============ STATE ============
  const [activeHistoryTab, setActiveHistoryTab] = useState('training');
  const [loading, setLoading] = useState(false);
  const [localTrainingHistory, setLocalTrainingHistory] = useState([]);
  const [localPredictionHistory, setLocalPredictionHistory] = useState([]);
  const [localModelComparison, setLocalModelComparison] = useState([]);

  // ============ LOCAL STORAGE ============
  useEffect(() => {
    const savedTraining = localStorage.getItem('trainingHistory');
    const savedPredictions = localStorage.getItem('predictionHistory');
    const savedComparison = localStorage.getItem('modelComparison');

    if (savedTraining) setLocalTrainingHistory(JSON.parse(savedTraining));
    if (savedPredictions) setLocalPredictionHistory(JSON.parse(savedPredictions));
    if (savedComparison) setLocalModelComparison(JSON.parse(savedComparison));
  }, []);

  useEffect(() => {
    if (trainingHistory && trainingHistory.length > 0) {
      localStorage.setItem('trainingHistory', JSON.stringify(trainingHistory));
      setLocalTrainingHistory(trainingHistory);
    }
    if (predictionHistory && predictionHistory.length > 0) {
      localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
      setLocalPredictionHistory(predictionHistory);
    }
    if (modelComparison && modelComparison.length > 0) {
      localStorage.setItem('modelComparison', JSON.stringify(modelComparison));
      setLocalModelComparison(modelComparison);
    }
  }, [trainingHistory, predictionHistory, modelComparison]);

  // ============ DATA LOADING ============
  const loadData = async () => {
    setLoading(true);
    try {
      if (activeHistoryTab === 'training' && loadTrainingHistory) {
        await loadTrainingHistory();
      } else if (activeHistoryTab === 'predictions' && loadPredictionHistory) {
        await loadPredictionHistory();
      } else if (activeHistoryTab === 'comparison' && loadModelComparison) {
        await loadModelComparison();
      }
    } catch (error) {
      console.error('Error loading data:', error);
      showMessage('error', 'Failed to load data from database');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [activeHistoryTab]);

  // ============ SAMPLE DATA ============
  const getSampleTrainingData = () => {
    if (trainingHistory.length > 0) return trainingHistory;
    if (localTrainingHistory.length > 0) return localTrainingHistory;
    return [
      { ModelName: 'KNN (Euclidean)', Accuracy: 0.823, Precision: 0.815, Recall: 0.820, F1Score: 0.818, TrainingDate: new Date().toISOString() },
      { ModelName: 'KNN (Manhattan)', Accuracy: 0.815, Precision: 0.808, Recall: 0.812, F1Score: 0.810, TrainingDate: new Date().toISOString() },
      { ModelName: 'Multinomial NB', Accuracy: 0.789, Precision: 0.782, Recall: 0.785, F1Score: 0.783, TrainingDate: new Date().toISOString() },
      { ModelName: 'RBF SVM', Accuracy: 0.856, Precision: 0.850, Recall: 0.853, F1Score: 0.851, TrainingDate: new Date().toISOString() },
      { ModelName: 'Neural Network', Accuracy: 0.867, Precision: 0.862, Recall: 0.865, F1Score: 0.863, TrainingDate: new Date().toISOString() },
      { ModelName: 'Decision Tree', Accuracy: 0.801, Precision: 0.795, Recall: 0.798, F1Score: 0.796, TrainingDate: new Date().toISOString() },
      { ModelName: 'Random Forest', Accuracy: 0.845, Precision: 0.840, Recall: 0.843, F1Score: 0.841, TrainingDate: new Date().toISOString() },
      { ModelName: 'Logistic Regression', Accuracy: 0.834, Precision: 0.828, Recall: 0.831, F1Score: 0.829, TrainingDate: new Date().toISOString() },
    ];
  };

  const getSamplePredictionData = () => {
    if (predictionHistory.length > 0) return predictionHistory;
    if (localPredictionHistory.length > 0) return localPredictionHistory;
    return [
      { PredictionDate: new Date().toISOString(), MidtermMarks: 18, SessionalMarks: 20, FinalExamMarks: 42, StudyHours: 6, PredictedGrade: 'A', Percentage: 80 },
      { PredictionDate: new Date(Date.now() - 86400000).toISOString(), MidtermMarks: 15, SessionalMarks: 15, FinalExamMarks: 35, StudyHours: 5, PredictedGrade: 'C', Percentage: 65 },
      { PredictionDate: new Date(Date.now() - 172800000).toISOString(), MidtermMarks: 22, SessionalMarks: 23, FinalExamMarks: 48, StudyHours: 8, PredictedGrade: 'A', Percentage: 93 },
      { PredictionDate: new Date(Date.now() - 259200000).toISOString(), MidtermMarks: 10, SessionalMarks: 12, FinalExamMarks: 28, StudyHours: 3, PredictedGrade: 'F', Percentage: 50 },
      { PredictionDate: new Date(Date.now() - 345600000).toISOString(), MidtermMarks: 16, SessionalMarks: 18, FinalExamMarks: 38, StudyHours: 5, PredictedGrade: 'B', Percentage: 72 },
    ];
  };

  const getSampleComparisonData = () => {
    if (modelComparison.length > 0) return modelComparison;
    if (localModelComparison.length > 0) return localModelComparison;
    return [
      { ModelName: 'Neural Network', Accuracy: 0.867, Precision: 0.862, Recall: 0.865, F1Score: 0.863, TrainingDate: new Date().toISOString() },
      { ModelName: 'RBF SVM', Accuracy: 0.856, Precision: 0.850, Recall: 0.853, F1Score: 0.851, TrainingDate: new Date().toISOString() },
      { ModelName: 'Random Forest', Accuracy: 0.845, Precision: 0.840, Recall: 0.843, F1Score: 0.841, TrainingDate: new Date().toISOString() },
      { ModelName: 'Logistic Regression', Accuracy: 0.834, Precision: 0.828, Recall: 0.831, F1Score: 0.829, TrainingDate: new Date().toISOString() },
      { ModelName: 'KNN (Euclidean)', Accuracy: 0.823, Precision: 0.815, Recall: 0.820, F1Score: 0.818, TrainingDate: new Date().toISOString() },
    ];
  };

  const trainingData = getSampleTrainingData();
  const predictionData = getSamplePredictionData();
  const comparisonData = getSampleComparisonData();

  // ============ HELPERS ============
  const getGradeBadge = (grade) => {
    const map = { 'A': 'badge-a', 'B': 'badge-b', 'C': 'badge-c', 'D': 'badge-d', 'F': 'badge-f' };
    return map[grade] || 'badge-info';
  };

  // ============ RENDER ============
  return (
    <div className="fade-in">
      <div className="card">
        <div className="card-title">
          <FaHistory />
          <span>Database History</span>
        </div>

        {!dbConnected && (
          <div className="alert alert-warning">
            <FaExclamationTriangle /> Database not connected. Showing sample data for demonstration.
          </div>
        )}

        <div className="tabs">
          <button
            className={`tab ${activeHistoryTab === 'training' ? 'active' : ''}`}
            onClick={() => setActiveHistoryTab('training')}
          >
            <FaChartBar /> Training History
          </button>
          <button
            className={`tab ${activeHistoryTab === 'predictions' ? 'active' : ''}`}
            onClick={() => setActiveHistoryTab('predictions')}
          >
            <FaTable /> Prediction History
          </button>
          <button
            className={`tab ${activeHistoryTab === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveHistoryTab('comparison')}
          >
            <FaAward /> Model Comparison
          </button>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <FaSpinner className="spinner" size={32} />
            <p style={{ marginTop: '16px', color: '#8b949e' }}>Loading data...</p>
          </div>
        ) : (
          <>
            {/* Training History */}
            {activeHistoryTab === 'training' && (
              <div className="table-scroll">
                {trainingData.length === 0 ? (
                  <div className="empty-state">
                    <FaDatabase size={32} style={{ color: '#484f58', marginBottom: '16px' }} />
                    <p>No training history found. Please train models first.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th><FaClock /> Training Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingData.map((record, idx) => (
                        <tr key={idx}>
                          <td className="model-name">{record.ModelName}</td>
                          <td>{(record.Accuracy * 100).toFixed(2)}%</td>
                          <td>{(record.Precision * 100).toFixed(2)}%</td>
                          <td>{(record.Recall * 100).toFixed(2)}%</td>
                          <td>{(record.F1Score * 100).toFixed(2)}%</td>
                          <td>{new Date(record.TrainingDate).toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}

            {/* Prediction History */}
            {activeHistoryTab === 'predictions' && (
              <div className="table-scroll">
                {predictionData.length === 0 ? (
                  <div className="empty-state">
                    <FaDatabase size={32} style={{ color: '#484f58', marginBottom: '16px' }} />
                    <p>No prediction history found. Make some predictions first.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th><FaClock /> Date</th>
                        <th>Midterm</th>
                        <th>Sessional</th>
                        <th>Final</th>
                        <th>Study Hours</th>
                        <th>Predicted Grade</th>
                        <th>Percentage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictionData.map((record, idx) => (
                        <tr key={idx}>
                          <td>{new Date(record.PredictionDate).toLocaleString()}</td>
                          <td>{record.MidtermMarks}</td>
                          <td>{record.SessionalMarks}</td>
                          <td>{record.FinalExamMarks}</td>
                          <td>{record.StudyHours}</td>
                          <td>
                            <span className={`badge ${getGradeBadge(record.PredictedGrade)}`}>
                              {record.PredictedGrade}
                            </span>
                          </td>
                          <td>{(record.Percentage || 0).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}

            {/* Model Comparison */}
            {activeHistoryTab === 'comparison' && (
              <div className="table-scroll">
                {comparisonData.length === 0 ? (
                  <div className="empty-state">
                    <FaDatabase size={32} style={{ color: '#484f58', marginBottom: '16px' }} />
                    <p>No model comparison data found.</p>
                  </div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Last Trained</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonData.map((record, idx) => (
                        <tr key={idx} style={idx === 0 ? { background: 'rgba(63, 185, 80, 0.05)' } : {}}>
                          <td className="model-name">
                            {record.ModelName}
                            {idx === 0 && (
                              <span style={{ marginLeft: '8px', fontSize: '11px', color: '#3fb950' }}>
                                <FaAward /> BEST
                              </span>
                            )}
                          </td>
                          <td style={{ fontWeight: '600', color: '#3fb950' }}>
                            {(record.Accuracy * 100).toFixed(2)}%
                          </td>
                          <td>{(record.Precision * 100).toFixed(2)}%</td>
                          <td>{(record.Recall * 100).toFixed(2)}%</td>
                          <td>{(record.F1Score * 100).toFixed(2)}%</td>
                          <td>{new Date(record.TrainingDate).toLocaleDateString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default History;
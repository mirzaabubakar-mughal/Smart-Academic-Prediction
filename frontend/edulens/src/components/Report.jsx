import React, { useState } from 'react';
import { 
  FaFileAlt, 
  FaDownload, 
  FaSpinner, 
  FaTrophy, 
  FaChartBar, 
  FaChartLine, 
  FaTable, 
  FaPrint,
  FaCheckCircle, 
  FaExclamationTriangle, 
  FaInfoCircle,
  FaGraduationCap, 
  FaDatabase, 
  FaBrain
} from 'react-icons/fa';
import { API_URL, ENDPOINTS, apiCall } from '../utils/api';

function Report({
  metrics,
  datasetInfo,
  modelsTrained,
  predictions,
  confusionMatrices,
  rocAucImages,
  graphImages,
  trainingStatus,
}) {
  // ============ STATE ============
  const [reportGenerated, setReportGenerated] = useState(false);
  const [bestModelMatrix, setBestModelMatrix] = useState(null);
  const [bestModelRoc, setBestModelRoc] = useState(null);
  const [bestModelName, setBestModelName] = useState('');
  const [bestModelAccuracy, setBestModelAccuracy] = useState(0);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  // ============ CONSTANTS ============
  const modelDisplayNames = {
    'knn_euclidean': 'KNN (Euclidean Distance)',
    'knn_manhattan': 'KNN (Manhattan Distance)',
    'multinomial_nb': 'Multinomial Naive Bayes',
    'svm_rbf': 'RBF Support Vector Machine',
    'neuralnetwork_softmax': 'Neural Network (Softmax)',
    'decision_tree': 'Decision Tree (CART)',
    'random_forest': 'Random Forest Ensemble',
    'logistic_regression': 'Logistic Regression',
  };

  // ============ LOAD BEST MODEL VISUALIZATIONS ============
  const loadBestModelVisualizations = async () => {
    if (!modelsTrained) return;
    setLoading(true);
    try {
      const [matrixResult, rocResult] = await Promise.all([
        apiCall(ENDPOINTS.VIZ_BEST_MATRIX),
        apiCall(ENDPOINTS.VIZ_BEST_ROC),
      ]);

      setBestModelMatrix(matrixResult.image);
      setBestModelRoc(rocResult.image);
      setBestModelName(matrixResult.model_name);

      if (metrics) {
        const best = Object.entries(metrics).reduce(
          (best, [name, m]) => m.accuracy > best.accuracy ? { name, accuracy: m.accuracy } : best,
          { name: '', accuracy: 0 }
        );
        setBestModelName(best.name);
        setBestModelAccuracy(best.accuracy);
      }
    } catch (err) {
      console.error('Error loading best model visualizations:', err);
    } finally {
      setLoading(false);
    }
  };

  // ============ HELPERS ============
  const safeGetValue = (obj, path, defaultValue) => {
    if (!obj) return defaultValue;
    const keys = path.split('.');
    let result = obj;
    for (let key of keys) {
      if (result === undefined || result === null) return defaultValue;
      result = result[key];
    }
    return result !== undefined && result !== null ? result : defaultValue;
  };

  const getGradeColor = (grade) => {
    const colors = { A: '#3fb950', B: '#58a6ff', C: '#d29922', D: '#f0883e', F: '#f85149' };
    return colors[grade] || '#8b949e';
  };

  // ============ REPORT GENERATION ============
  const generateReport = () => {
    setReportGenerated(true);
    loadBestModelVisualizations();
  };

  const downloadPDF = async () => {
    setDownloading(true);
    try {
      if (typeof html2pdf === 'undefined') {
        console.warn('html2pdf not loaded, using fallback method');
        window.print();
        setDownloading(false);
        return;
      }

      const reportContent = document.getElementById('report-content');
      if (!reportContent) return;

      const reportClone = reportContent.cloneNode(true);
      const images = reportClone.querySelectorAll('img');

      await Promise.all(Array.from(images).map(img => {
        if (img.complete) return Promise.resolve();
        return new Promise((resolve) => {
          img.onload = resolve;
          img.onerror = resolve;
          setTimeout(resolve, 1000);
        });
      }));

      const options = {
        margin: [10, 10, 10, 10],
        filename: `academic_performance_report_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: {
          scale: 2,
          logging: false,
          useCORS: true,
          letterRendering: true,
        },
        jsPDF: {
          unit: 'mm',
          format: 'a4',
          orientation: 'portrait',
        },
      };

      await html2pdf().set(options).from(reportClone).save();
    } catch (error) {
      console.error('Error generating PDF:', error);
      window.print();
    } finally {
      setDownloading(false);
    }
  };

  const totalSamples = safeGetValue(datasetInfo, 'shape.0', 2000);
  const totalFeatures = safeGetValue(datasetInfo, 'shape.1', 8);

  // ============ RENDER ============
  return (
    <div className="fade-in">
      {/* Report Generation Card */}
      <div className="card">
        <div className="card-title">
          <FaFileAlt />
          <span>Generate Professional Report</span>
        </div>
        <p className="card-subtitle">
          Generate a professional report including dataset overview, model performance metrics,
          best model analysis, confusion matrix, ROC curves, and predictions.
        </p>
        <button className="btn-primary" onClick={generateReport} disabled={!modelsTrained || loading}>
          {loading ? <FaSpinner className="spinner" /> : <FaFileAlt />} 
          {loading ? 'Loading Visualizations...' : 'Generate Report'}
        </button>
      </div>

      {/* Report Content */}
      {reportGenerated && modelsTrained && (
        <div>
          <div className="report-container" id="report-content" style={{ background: '#161b22', padding: '40px', borderRadius: '12px', border: '1px solid #30363d' }}>
            {/* Header */}
            <div style={{ textAlign: 'center', marginBottom: '30px', paddingBottom: '20px', borderBottom: '2px solid #58a6ff' }}>
              <h1 style={{ color: '#58a6ff', fontSize: '28px', marginBottom: '8px' }}>Academic Performance Report</h1>
              <p style={{ color: '#8b949e', fontSize: '14px' }}>Generated on: {new Date().toLocaleString()}</p>
              <p style={{ color: '#3fb950', fontSize: '12px', marginTop: '8px' }}>Powered by EduLens Academic Performance System</p>
            </div>

            {/* Executive Summary */}
            <div style={{ marginBottom: '30px', padding: '20px', background: 'rgba(88, 166, 255, 0.05)', borderRadius: '12px' }}>
              <h2 style={{ color: '#58a6ff', fontSize: '20px', marginBottom: '16px' }}>Executive Summary</h2>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <div style={{ textAlign: 'center', padding: '16px', background: '#1c2333', borderRadius: '8px', border: '1px solid #30363d' }}>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#58a6ff' }}>{totalSamples}</div>
                  <div style={{ fontSize: '12px', color: '#8b949e' }}>Total Samples</div>
                </div>
                <div style={{ textAlign: 'center', padding: '16px', background: '#1c2333', borderRadius: '8px', border: '1px solid #30363d' }}>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#3fb950' }}>{totalFeatures}</div>
                  <div style={{ fontSize: '12px', color: '#8b949e' }}>Total Features</div>
                </div>
                <div style={{ textAlign: 'center', padding: '16px', background: '#1c2333', borderRadius: '8px', border: '1px solid #30363d' }}>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#d29922' }}>{metrics ? Object.keys(metrics).length : 8}</div>
                  <div style={{ fontSize: '12px', color: '#8b949e' }}>Models Trained</div>
                </div>
                <div style={{ textAlign: 'center', padding: '16px', background: '#1c2333', borderRadius: '8px', border: '1px solid #30363d' }}>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#f85149' }}>
                    {bestModelAccuracy ? (bestModelAccuracy * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                  <div style={{ fontSize: '12px', color: '#8b949e' }}>Best Model Accuracy</div>
                </div>
              </div>
            </div>

            {/* Dataset Overview */}
            <div style={{ marginBottom: '30px' }}>
              <h2 style={{ color: '#58a6ff', fontSize: '20px', marginBottom: '16px', borderLeft: '4px solid #58a6ff', paddingLeft: '12px' }}>Dataset Overview</h2>
              <table style={{ width: '100%', borderCollapse: 'collapse', background: '#1c2333', borderRadius: '8px', overflow: 'hidden' }}>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #30363d' }}>
                    <td style={{ padding: '12px', fontWeight: '600', width: '200px' }}>Total Dataset Samples</td>
                    <td style={{ padding: '12px' }}>{totalSamples}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #30363d' }}>
                    <td style={{ padding: '12px', fontWeight: '600' }}>Training Samples (80%)</td>
                    <td style={{ padding: '12px' }}>{Math.round(totalSamples * 0.8)}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #30363d' }}>
                    <td style={{ padding: '12px', fontWeight: '600' }}>Testing Samples (20%)</td>
                    <td style={{ padding: '12px' }}>{Math.round(totalSamples * 0.2)}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #30363d' }}>
                    <td style={{ padding: '12px', fontWeight: '600' }}>Total Features</td>
                    <td style={{ padding: '12px' }}>{totalFeatures}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #30363d' }}>
                    <td style={{ padding: '12px', fontWeight: '600' }}>Target Variable</td>
                    <td style={{ padding: '12px' }}>Grade (A, B, C, D, F)</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '12px', fontWeight: '600' }}>Number of Classes</td>
                    <td style={{ padding: '12px' }}>5 (A, B, C, D, F)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Best Model Analysis */}
            <div style={{ marginBottom: '30px' }}>
              <h2 style={{ color: '#58a6ff', fontSize: '20px', marginBottom: '16px', borderLeft: '4px solid #58a6ff', paddingLeft: '12px' }}>Best Model Analysis</h2>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px', background: 'rgba(210, 153, 34, 0.1)', borderRadius: '12px', marginBottom: '20px', flexWrap: 'wrap', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '14px', color: '#8b949e' }}>Best Performing Model</div>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#e6edf3' }}>{modelDisplayNames[bestModelName] || bestModelName}</div>
                </div>
                <div>
                  <div style={{ fontSize: '14px', color: '#8b949e' }}>Accuracy Score</div>
                  <div style={{ fontSize: '32px', fontWeight: '700', color: '#3fb950' }}>{(bestModelAccuracy * 100).toFixed(2)}%</div>
                </div>
              </div>

              {bestModelMatrix && (
                <div style={{ marginBottom: '20px' }}>
                  <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px', color: '#e6edf3' }}>Confusion Matrix</h3>
                  <img src={bestModelMatrix} style={{ width: '100%', maxWidth: '600px', borderRadius: '8px', border: '1px solid #30363d', margin: '0 auto', display: 'block' }} />
                  <p style={{ fontSize: '12px', color: '#8b949e', textAlign: 'center', marginTop: '8px' }}>
                    Diagonal cells show correct predictions, off-diagonal show misclassifications
                  </p>
                </div>
              )}

              {bestModelRoc && (
                <div style={{ marginBottom: '20px' }}>
                  <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '12px', color: '#e6edf3' }}>ROC/AUC Curve</h3>
                  <img src={bestModelRoc} style={{ width: '100%', maxWidth: '600px', borderRadius: '8px', border: '1px solid #30363d', margin: '0 auto', display: 'block' }} />
                  <p style={{ fontSize: '12px', color: '#8b949e', textAlign: 'center', marginTop: '8px' }}>
                    ROC curves show model performance across different thresholds. Higher AUC indicates better performance.
                  </p>
                </div>
              )}
            </div>

            {/* All Models Performance */}
            <div style={{ marginBottom: '30px' }}>
              <h2 style={{ color: '#58a6ff', fontSize: '20px', marginBottom: '16px', borderLeft: '4px solid #58a6ff', paddingLeft: '12px' }}>All Models Performance Comparison</h2>
              <table style={{ width: '100%', borderCollapse: 'collapse', background: '#1c2333', borderRadius: '8px', overflow: 'hidden' }}>
                <thead>
                  <tr style={{ background: '#58a6ff', color: '#0d1117' }}>
                    <th style={{ padding: '12px', textAlign: 'left' }}>Model</th>
                    <th style={{ padding: '12px', textAlign: 'center' }}>Accuracy</th>
                    <th style={{ padding: '12px', textAlign: 'center' }}>Precision</th>
                    <th style={{ padding: '12px', textAlign: 'center' }}>Recall</th>
                    <th style={{ padding: '12px', textAlign: 'center' }}>F1-Score</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics && Object.entries(metrics)
                    .sort((a, b) => b[1].accuracy - a[1].accuracy)
                    .map(([model, m], idx) => (
                      <tr key={model} style={{ borderBottom: '1px solid #30363d', background: idx === 0 ? 'rgba(63, 185, 80, 0.05)' : 'transparent' }}>
                        <td style={{ padding: '12px', fontWeight: '500' }}>
                          {idx === 0 && <span style={{ marginRight: '8px' }}>🏆</span>}
                          {modelDisplayNames[model] || model}
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center', fontWeight: '600', color: idx === 0 ? '#3fb950' : '#e6edf3' }}>
                          {(m.accuracy * 100).toFixed(2)}%
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>{(m.precision * 100).toFixed(2)}%</td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>{(m.recall * 100).toFixed(2)}%</td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>{(m.f1 * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>

            {/* Latest Prediction */}
            {predictions && (
              <div style={{ marginBottom: '30px' }}>
                <h2 style={{ color: '#58a6ff', fontSize: '20px', marginBottom: '16px', borderLeft: '4px solid #58a6ff', paddingLeft: '12px' }}>Latest Prediction</h2>
                <div style={{ padding: '20px', borderRadius: '12px', border: `2px solid ${getGradeColor(predictions.final_prediction)}`, textAlign: 'center' }}>
                  <div style={{ fontSize: '48px', fontWeight: '800', color: getGradeColor(predictions.final_prediction) }}>
                    {predictions.final_prediction}
                  </div>
                  <div style={{ fontSize: '18px', fontWeight: '600', marginTop: '8px' }}>Predicted Grade</div>
                  <div style={{ marginTop: '12px', color: '#8b949e' }}>Score: {(predictions.total_marks || 0).toFixed(1)}/100</div>
                  <div style={{ width: '200px', height: '8px', background: '#161b22', borderRadius: '4px', margin: '8px auto', overflow: 'hidden' }}>
                    <div style={{ width: `${(predictions.percentage || 0)}%`, height: '100%', background: getGradeColor(predictions.final_prediction) }} />
                  </div>
                  <div style={{ marginTop: '16px', textAlign: 'left', padding: '16px', background: '#1c2333', borderRadius: '8px' }}>
                    <div style={{ fontWeight: '600', marginBottom: '8px' }}>Remarks</div>
                    <div style={{ fontSize: '14px', color: '#8b949e' }}>{predictions.remarks?.detailed || ''}</div>
                    <div style={{ marginTop: '12px', padding: '12px', background: '#161b22', borderRadius: '8px' }}>
                      <div style={{ fontWeight: '600', marginBottom: '8px' }}>Improvement Suggestions</div>
                      <div style={{ fontSize: '13px', color: '#8b949e' }}>{predictions.remarks?.improvement || ''}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Footer */}
            <div style={{ marginTop: '30px', paddingTop: '20px', borderTop: '2px solid #30363d', textAlign: 'center' }}>
              <p style={{ fontSize: '12px', color: '#8b949e' }}>
                This report was generated automatically and the predictions are based on Supervised ML models trained on student performance data.
              </p>
              <p style={{ fontSize: '11px', color: '#484f58', marginTop: '8px' }}>
                © {new Date().getFullYear()} EduLens - AI-Powered Academic Performance System
              </p>
            </div>
          </div>

          {/* Download Buttons */}
          <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginTop: '20px', flexWrap: 'wrap' }}>
            <button onClick={downloadPDF} disabled={downloading} className="btn-primary">
              {downloading ? <FaSpinner className="spinner" /> : <FaDownload />} 
              {downloading ? 'Generating PDF...' : 'Download Report'}
            </button>
            <button onClick={() => window.print()} className="btn-secondary">
              <FaPrint /> Print Report
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Report;
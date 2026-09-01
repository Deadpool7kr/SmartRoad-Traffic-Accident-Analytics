from pathlib import Path
import sys, unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from analytics import load_data, preprocess, clean_numeric_ranges, build_model, fit_cluster

DATA=Path(__file__).resolve().parents[1]/'data'/'demo_accidents.csv'

class TestSmartRoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=load_data(DATA); cls.df=clean_numeric_ranges(preprocess(cls.raw))
    def test_duplicates_removed_by_preprocess(self):
        self.assertLess(len(self.df),len(self.raw))
    def test_engineered_columns(self):
        for c in ['hour','month','weekend','night','rush_hour','severity_label','high_severity']:
            self.assertIn(c,self.df.columns)
    def test_three_model_comparison(self):
        import pandas as pd
        from src.analytics import load_data, preprocess, build_model
        root=Path(__file__).resolve().parents[1]
        df=preprocess(load_data(root / "data" / "demo_accidents.csv"))
        for name in ["Random Forest", "Logistic Regression", "Gradient Boosting"]:
            _,_,_,_,metrics,_=build_model(df, name)
            self.assertGreaterEqual(metrics["cv_f1_mean"], 0.0)
            self.assertLessEqual(metrics["cv_f1_mean"], 1.0)

    def test_model_metrics(self):
        pipe,Xt,yt,pred,metrics,features=build_model(self.df,'Random Forest')
        for k in ['accuracy','precision','recall','f1','cv_f1_mean']:
            self.assertGreaterEqual(metrics[k],0.0); self.assertLessEqual(metrics[k],1.0)
        self.assertGreater(metrics['f1'],0.0)
        self.assertGreater(metrics['cv_f1_mean'],0.0)
    def test_confusion_matrix_shape(self):
        from sklearn.metrics import confusion_matrix
        pipe,Xt,yt,pred,metrics,features=build_model(self.df,'Logistic Regression')
        cm=confusion_matrix(yt,pred)
        self.assertEqual(cm.shape,(2,2))
        self.assertEqual(int(cm.sum()),len(yt))

    def test_model_comparison_outputs(self):
        for name in ['Random Forest','Logistic Regression','Gradient Boosting']:
            pipe,Xt,yt,pred,metrics,features=build_model(self.df,name)
            for k in ['accuracy','precision','recall','f1','cv_f1_mean','cv_f1_std']:
                self.assertIn(k,metrics)
                self.assertGreaterEqual(metrics[k],0.0)
                self.assertLessEqual(metrics[k],1.0)

    def test_cluster(self):
        out,summ=fit_cluster(self.df,8)
        self.assertEqual(summ['cluster'].nunique(),8)
        self.assertEqual(len(out), self.df[['latitude','longitude']].dropna().shape[0])

if __name__=='__main__': unittest.main()

class TestAutomatedReport(unittest.TestCase):
    def test_report_generator(self):
        from src.report_generator import generate_pdf_report
        import tempfile
        root = Path(__file__).resolve().parents[1]
        summary = root / 'outputs' / 'run_summary.json'
        charts = root / 'outputs'
        self.assertTrue(summary.exists())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'report.pdf'
            result = generate_pdf_report(summary, out, charts)
            self.assertTrue(result.exists())
            self.assertGreater(result.stat().st_size, 10000)

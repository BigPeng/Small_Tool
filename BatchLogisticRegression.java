import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 分快梯度下降训练逻辑回归
 * 
 * @author hadoop
 * 
 */
public class BatchLogisticRegression {
	// 数值型数据下标
	static HashSet<Integer> realAttrIndex;
	private static final int VEC_MENSION = 426757 + 1 + 142573;
	// 记录个数
	// private static final int RECORD_NUM = 45840617;
	// private static final int RECORD_NUM = 200000;
	// 学习速率

	// private static double[] theta =
	// initTheta(VEC_MENSION);
	private static int[] offset = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13,
			1474, 2058, 2364, 2389, 14907, 15541, 15545, 108691, 114375,
			117570, 117598, 132591, 132602, 138255, 140429, 140434, 140453,
			140469, 426651, 426757 };
	private ExecutorService exec;
	private int cpuNum;

	public BatchLogisticRegression(int cpuNum) {
		this.cpuNum = cpuNum;
		exec = Executors.newFixedThreadPool(cpuNum);
	}

	/**
	 * 梯度下降求一个batch的梯度
	 * 
	 * @param batch
	 * @param theta
	 * @return
	 */
	private Holder gredaint(List<double[]> batch, double[] theta) {
		double[] deltas = new double[theta.length];
		double error = 0;
		for (double[] record : batch) {
			double hx = theta[theta.length - 1];// theta0
			for (int i = 0; i < record.length - 1; i++) {
				if (realAttrIndex.contains(i)) {
					hx += record[i] * theta[i];
				} else {
					int index = (int) (record[i]);
					if (index != 0) {// 忽略缺失值
						try {
							hx += theta[index + offset[i]];
						} catch (Exception e) {
							System.out.println(Arrays.toString(record));
							System.out.println(Arrays.toString(offset));
							System.out.println("i:" + i + ", index:" + index
									+ "  offset[i]" + offset[i]);
							throw new RuntimeException(e);
						}
					}
				}
			}
			hx = 1.0 / (1 + Math.exp(-hx));
			double y = record[record.length - 1];
			if (t++ % 85000000 == 0)
				System.out.println(y + " :" + hx);
			double delta = y - hx;

			for (int i = 0; i < record.length - 1; i++) {
				if (realAttrIndex.contains(i)) {
					deltas[i] += record[i] * delta;
				} else {
					int index = (int) (record[i]);
					if (index != 0) {// 忽略缺失值
						deltas[index + offset[i]] += delta;
					}
				}
			}
			deltas[theta.length - 1] += delta;// delta0
			error += Math.abs(delta);
		}
		Holder h = new Holder(deltas, error);
		return h;
	}

	/**
	 * 更新参赛
	 * 
	 * @param theta
	 * @param deltas
	 * @param alpha
	 * @param m
	 */
	private int t = 1;

	private void update(double[] theta, double[] deltas, double alpha, int m) {

		// System.out.println(Arrays.toString(theta));
		for (int i = 0; i < theta.length; i++) {
			theta[i] = theta[i] * 0.9999999 + deltas[i] * alpha / m;
		}

	}

	private double dispath(List<double[]> subset, final double[] theta,
			double alpha) throws InterruptedException, ExecutionException {
		int batchSize = (subset.size() + cpuNum - 1) / cpuNum;// 向上取整
		if (batchSize < 10) {// 数据太少了，一个线程负责就好了
			Holder holder = gredaint(subset, theta);
			update(theta, holder.deltas, alpha, subset.size());
			return holder.error;
		} else {// 多线程处理
			CompletionService<Holder> completeService = new ExecutorCompletionService<>(
					exec);
			for (int i = 0; i < cpuNum; i++) {
				int end = Math.min((i + 1) * batchSize, subset.size());
				final List<double[]> batch = subset.subList(i * batchSize, end);
				completeService.submit(new Callable<Holder>() {

					@Override
					public Holder call() throws Exception {
						return gredaint(batch, theta);
					}
				});
			}
			double[] deltas = new double[theta.length];
			double error = 0;
			for (int i = 0; i < cpuNum; i++) {
				Holder holder = completeService.take().get();// 等待处理结果
				for (int j = 0; j < deltas.length; j++) {
					deltas[j] += holder.deltas[j];
				}
				error += holder.error;
			}
			error = error / subset.size();
			// System.out.println(error);
			// 所有线程跑完了，可以更新参数了
			update(theta, deltas, alpha, subset.size());
			return error;
		}

	}

	/**
	 * 扫描文件，读入一定量记录后，进行批量训练
	 * 
	 * @param fileName
	 * @param theta
	 * @param alpha
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	public double train(String fileName, int subSize, double[] theta,
			double alpha) throws IOException, InterruptedException,
			ExecutionException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fileName)));
		List<double[]> subset = new ArrayList<double[]>();
		int lineCount = 0;
		double error = 0;
		int batchCount = 0;
		while (true) {
			String line = null;
			line = reader.readLine();
			if (line == null) {
				if (subset.size() > 0) {
					error += dispath(subset, theta, alpha);
					batchCount++;
				}
				break;
			}
			String[] s = line.trim().split(",");
			double[] record = new double[s.length];
			for (int i = 0; i < record.length; i++) {
				if (s[i].equals("\\N"))// 缺失值
					record[i] = 0;
				else {
					record[i] = Double.valueOf(s[i]);
					if (needNormal == true && i == 5)
						record[i] /= 5295.0;
				}
			}
			subset.add(record);
			lineCount++;
			if (lineCount % subSize == 0) {
				error += dispath(subset, theta, alpha);
				subset.clear();
				batchCount++;
			}
		}
		reader.close();
		return error / batchCount;
	}

	private double[] initTheta(int n) {
		double[] theta = new double[n];
		Random r = new Random(1);
		for (int i = 0; i < n; i++) {
			theta[i] = (r.nextDouble() - 0.5);
		}
		return theta;
	}

	private void savePara(double[] theta, String index)
			throws FileNotFoundException {
		PrintWriter out = new PrintWriter("data/theta_" + index + ".txt");
		for (int i = 0; i < theta.length; i++) {
			out.write(theta[i] + "\n");
		}
		out.flush();
		out.close();
	}

	private double[] initThetaFromFile(String fileName, int n)
			throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fileName)));
		double[] theta = new double[n];
		for (int i = 0; i < n; i++) {
			String line = reader.readLine();
			theta[i] = Double.valueOf(line);
		}
		reader.close();
		return theta;
	}

	static class Holder {
		public final double[] deltas;
		public double error;

		public Holder(double[] deltas, double error) {
			this.deltas = deltas;
			this.error = error;
		}
	}

	/**
	 * 
	 * @param record
	 * @param theta
	 * @return
	 */
	private double predict(double[] record, double[] theta) {
		double hx = theta[theta.length - 1];// theta0
		for (int i = 0; i < record.length - 1; i++) {
			if (realAttrIndex.contains(i)) {
				hx += record[i] * theta[i];
			} else {
				int index = (int) (record[i]);
				if (index != 0) {// 忽略缺失值
					hx += theta[index + offset[i]];
				}
			}
		}
		hx = 1.0 / (1 + Math.exp(-hx));
		// System.out.println(hx);
		return hx;
	}

	private void predict(String fileName, String outFileName, double[] theta)
			throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				new FileInputStream(fileName)));
		PrintWriter out = new PrintWriter(outFileName);

		String line = reader.readLine();
		int count = 0;
		while (line != null) {
			String[] s = line.trim().split(",");
			double[] record = new double[s.length - 1];
			for (int i = 0; i < s.length - 1; i++) {
				if (s[i].equals("\\N"))
					record[i] = 0;
				else {
					record[i] = Double.valueOf(s[i]);
					if (needNormal == true && i == 5)
						record[i] /= 5295.0;
				}
			}
			double hx = predict(record, theta);
			String d = String.format("%.8f", hx);
			out.write(s[s.length - 1] + "," + d + "\n");
			line = reader.readLine();
			if (++count % 200000 == 0)
				System.out.println(count);
		}
		out.flush();
		out.close();
		reader.close();
	}

	private static void predict() throws IOException {
		BatchLogisticRegression blr = new BatchLogisticRegression(4);
		double[] theta = blr
				.initThetaFromFile("data/theta_99.txt", VEC_MENSION);
		blr.predict("E://dataset//CTR//format_test_normal.csv",
				"E://dataset//CTR//predict.csv", theta);
	}

	private static void train() throws Exception {
		BatchLogisticRegression blr = new BatchLogisticRegression(4);
		String fileName = "E://dataset//CTR//format_train_normal.csv";
		// double[] theta = blr.initTheta(VEC_MENSION);
		double[] theta = blr
				.initThetaFromFile("data/theta_99.txt", VEC_MENSION);
		int iter = 100;// 迭代次数
		for (int i = 0; i < iter; i++) {
			double error = blr.train(fileName, 400000, theta, 0.04);
			System.out.println(i + "th error:" + error);
			blr.savePara(theta, i + "");
		}
	}

	static void minitestTrain() throws Exception {
		offset = new int[] { 20, 0, 0 };
		realAttrIndex = new HashSet<Integer>();
		int vecDemension = 0;
		for (int i = 0; i < offset.length; i++) {
			if (offset[i] == 0) {
				realAttrIndex.add(i);
				vecDemension += 1;
			} else
				vecDemension += offset[i];
		}
		System.out.println(vecDemension);
		offset = new int[] { 0, 20, 21 };
		BatchLogisticRegression blr = new BatchLogisticRegression(4);
		double[] theta = blr.initTheta(vecDemension + 1);
		int iter = 1000;// 迭代次数
		String fileName = "E://BaiduDrive//百度云同步盘//backup//datamining//ccfcontest//has_user_info//mini_data.tmp";
		for (int i = 0; i < iter; i++) {
			double error = blr.train(fileName, 50, theta, 0.1);
			System.out.println(i + ":" + error);

		}
		blr.savePara(theta, iter + "");
	}

	static void minitestPredict() throws Exception {
		offset = new int[] { 20, 0, 0 };
		realAttrIndex = new HashSet<Integer>();
		int vecDemension = 0;
		for (int i = 0; i < offset.length; i++) {
			if (offset[i] == 0) {
				realAttrIndex.add(i);
				vecDemension += 1;
			} else
				vecDemension += offset[i];
		}
		System.out.println(vecDemension);
		offset = new int[] { 0, 20, 21 };
		BatchLogisticRegression blr = new BatchLogisticRegression(4);
		double[] theta = blr.initThetaFromFile("data/theta_1000.txt",
				vecDemension + 1);
		String fileName = "E://BaiduDrive//百度云同步盘//backup//datamining//ccfcontest//has_user_info//mini_data.tmp";
		String out = "E://BaiduDrive//百度云同步盘//backup//datamining//ccfcontest//has_user_info//test_out.tmp";
		blr.predict(fileName, out, theta);
	}

	private static boolean needNormal = false;

	static void ccfTrain() throws Exception {
		// img_popu,
		// adv_topic,imgid,imgclass,userinfo,bright,lable
		needNormal = true;
		int[] count = new int[] { 1, 30, 21, 3, 71, 1, 35, 1489, 1 };
		offset = new int[count.length];
		realAttrIndex = new HashSet<Integer>();
		int vecDemension = 0;
		for (int i = 0; i < offset.length; i++) {
			if (count[i] == 1)
				realAttrIndex.add(i);
			if (i > 0)
				offset[i] = offset[i - 1] + count[i - 1];
			vecDemension += count[i];
		}
		System.out.println(vecDemension);
		System.out.println(Arrays.toString(offset));

		BatchLogisticRegression blr = new BatchLogisticRegression(4);
		 double[] theta = blr.initTheta(vecDemension
		 + 1);
//		double[] theta = blr.initThetaFromFile(
//				"data/theta_trainall_imgclass_nocolor_advtopicOne_popu70.txt",
//				vecDemension + 1);
		int iter = 4;// 迭代次数
		String fileName = "E://BaiduDrive//百度云同步盘//backup//datamining//ccfcontest//has_user_info//trainall_imgclass_nocolor_advtopicOne_popu.tmp";
		for (int i = 0; i < iter; i++) {
			System.out.print(i + "th error:");
			double error = blr.train(fileName, 5000, theta, 0.1);
			System.out.println(error);
			if (i % 3 == 0)
				blr.savePara(theta,
						"trainall_imgclass_nocolor_advtopicOne_popu_alpha" + i);
		}
		blr.exec.shutdown();
	}

	static void ccfPredict() throws Exception {
		needNormal = true;
		int[] count = new int[] { 1, 30, 21, 3, 71, 1, 35, 1489, 1 };
		offset = new int[count.length];
		realAttrIndex = new HashSet<Integer>();
		int vecDemension = 0;
		for (int i = 0; i < offset.length; i++) {
			if (count[i] == 1)
				realAttrIndex.add(i);
			if (i > 0)
				offset[i] = offset[i - 1] + count[i - 1];
			vecDemension += count[i];
		}
		System.out.println(vecDemension);
		System.out.println(Arrays.toString(offset));
		BatchLogisticRegression blr = new BatchLogisticRegression(4);
		double[] theta = blr.initThetaFromFile(
				"data/theta_trainall_imgclass_nocolor_advtopicOne_popu_alpha3.txt",
				vecDemension + 1);
		String fileName = "E://BaiduDrive//百度云同步盘//backup//datamining//ccfcontest//has_user_info//test_imgclass_nocolor_advtopicOne_popu.tmp";
		String out = "E://BaiduDrive//百度云同步盘//backup//datamining//ccfcontest//has_user_info//test_imgclass_out.tmp";
		blr.predict(fileName, out, theta);
		blr.exec.shutdown();
	}

	public static void main(String[] args) throws Exception {
		// train();
		// predict();06899057805460887
		//ccfTrain();
		ccfPredict();
		// minitestTrain();
		// minitestPredict();
	}
}

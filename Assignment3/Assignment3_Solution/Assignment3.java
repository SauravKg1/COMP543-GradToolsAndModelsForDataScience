import java.io.IOException;
import java.util.PriorityQueue;
import java.util.StringTokenizer;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class Assignment3 extends Configured implements Tool {
    private static String[] splitCSV(String line) {
        return line.split(",", -1); // keep empties; we validate length & numeric
    }

    private static boolean isHeader(String[] f) {
        if (f.length < 11) return false;
        // header likely has literal column names; check a few
        return "medallion".equalsIgnoreCase(f[0]) || "hack_license".equalsIgnoreCase(f[1])
                || "total_amount".equalsIgnoreCase(f[10]);
    }

    private static String extractDate(String pickupDateTime) {
        // Expect "YYYY-MM-DD HH:MM:SS"; return date part
        if (pickupDateTime == null || pickupDateTime.length() < 10) return null;
        return pickupDateTime.substring(0, 10);
    }

    private static Double parseDoubleSafe(String s) {
        try {
            return Double.parseDouble(s);
        } catch (Exception e) {
            return null;
        }
    }

    // -----------------------------
    // Task 1: Revenue per date
    // -----------------------------

    public static class DateRevenueMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        private final Text outKey = new Text();
        private final DoubleWritable outVal = new DoubleWritable();

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            final String line = value.toString();
            if (line == null || line.isEmpty()) return;

            final String[] f = splitCSV(line);
            if (f.length < 11) return;                  // skip short/dirty
            if (isHeader(f)) return;                    // skip header

            final String date = extractDate(f[3]);
            if (date == null) return;

            final Double total = parseDoubleSafe(f[10]);
            if (total == null) return;

            // Optional month filter via -D filter.month=YYYY-MM (only emit those dates if provided)
            String monthFilter = ctx.getConfiguration().get("filter.month", "").trim();
            if (!monthFilter.isEmpty() && !date.startsWith(monthFilter)) return;

            outKey.set(date);
            outVal.set(total);
            ctx.write(outKey, outVal);
        }
    }

    public static class SumDoubleReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        private final DoubleWritable out = new DoubleWritable();

        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> vals, Context ctx)
                throws IOException, InterruptedException {
            double sum = 0.0;
            for (DoubleWritable v : vals) {
                sum += v.get();
            }
            out.set(sum);
            ctx.write(key, out);
        }
    }

    // Safe to reuse reducer as combiner for Task 1 (summing doubles)
    public static class SumDoubleCombiner extends SumDoubleReducer { }

    // --------------------------------------
    // Task 2: Job A -> revenue per driver
    // --------------------------------------

    public static class DriverRevenueMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
        private final Text outKey = new Text();
        private final DoubleWritable outVal = new DoubleWritable();

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            final String line = value.toString();
            if (line == null || line.isEmpty()) return;

            final String[] f = splitCSV(line);
            if (f.length < 11) return;
            if (isHeader(f)) return;

            final String driver = f[1];
            if (driver == null || driver.isEmpty()) return;

            final Double total = parseDoubleSafe(f[10]);
            if (total == null) return;

            outKey.set(driver);
            outVal.set(total);
            ctx.write(outKey, outVal);
        }
    }

    public static class DriverRevenueReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        private final DoubleWritable out = new DoubleWritable();
        @Override
        protected void reduce(Text key, Iterable<DoubleWritable> vals, Context ctx)
                throws IOException, InterruptedException {
            double sum = 0.0;
            for (DoubleWritable v : vals) sum += v.get();
            out.set(sum);
            ctx.write(key, out); // (driver, totalRevenue)
        }
    }

    // --------------------------------------
    // Task 2: Job B -> top-10 via PQ
    // Mapper: keep local top-10; emit only those in cleanup with SAME KEY to force a single reducer
    // Reducer: aggregate top-10 of mapper outputs and emit final top-10
    // --------------------------------------

    // Simple pair holder (use Java types, not Hadoop writables, per instructions)
    private static class DriverRev {
        final String driver;
        final double revenue;
        DriverRev(String d, double r) { this.driver = d; this.revenue = r; }
    }

    public static class Top10Mapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        private static final int K = 10;
        private final PriorityQueue<DriverRev> pq = new PriorityQueue<>(Comparator.comparingDouble(dr -> dr.revenue));
        private final IntWritable ONE_KEY = new IntWritable(1);
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context ctx) throws IOException, InterruptedException {
            // Each line from Job A looks like: "<driver>\t<sumRevenue>"
            final String line = value.toString();
            if (line == null || line.isEmpty()) return;

            int tab = line.lastIndexOf('\t');
            if (tab <= 0 || tab >= line.length() - 1) {
                // try space split fallback
                String[] parts = line.trim().split("\\s+");
                if (parts.length < 2) return;
                addCandidate(parts[0], parts[parts.length - 1]);
            } else {
                String driver = line.substring(0, tab);
                String revStr = line.substring(tab + 1);
                addCandidate(driver, revStr);
            }
        }

        private void addCandidate(String driver, String revStr) {
            Double rev = parseDoubleSafe(revStr);
            if (rev == null) return;

            // Maintain min-heap of size <= 10
            if (pq.size() < K) {
                pq.add(new DriverRev(driver, rev));
            } else if (pq.peek().revenue < rev) {
                pq.poll();
                pq.add(new DriverRev(driver, rev));
            }
        }

        @Override
        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            // Emit only the mapper's top-10
            // We emit as "driver\trevenue"
            for (DriverRev dr : pq) {
                outVal.set(dr.driver + "\t" + dr.revenue);
                ctx.write(ONE_KEY, outVal);
            }
        }
    }

    public static class Top10Reducer extends Reducer<IntWritable, Text, Text, DoubleWritable> {
        private static final int K = 10;
        private final PriorityQueue<DriverRev> pq = new PriorityQueue<>(Comparator.comparingDouble(dr -> dr.revenue));
        private final Text outKey = new Text();
        private final DoubleWritable outVal = new DoubleWritable();

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context ctx)
                throws IOException, InterruptedException {
            for (Text t : values) {
                String line = t.toString();
                int tab = line.lastIndexOf('\t');
                if (tab <= 0 || tab >= line.length() - 1) continue;
                String driver = line.substring(0, tab);
                Double rev = parseDoubleSafe(line.substring(tab + 1));
                if (rev == null) continue;

                if (pq.size() < K) {
                    pq.add(new DriverRev(driver, rev));
                } else if (pq.peek().revenue < rev) {
                    pq.poll();
                    pq.add(new DriverRev(driver, rev));
                }
            }
        }

        @Override
        protected void cleanup(Context ctx) throws IOException, InterruptedException {
            // pq is min-heap; to output highest-first, collect then sort descending
            PriorityQueue<DriverRev> max = new PriorityQueue<>(Comparator.comparingDouble((DriverRev dr) -> dr.revenue).reversed());
            max.addAll(pq);
            int rank = 1;
            while (!max.isEmpty()) {
                DriverRev dr = max.poll();
                outKey.set(rank + "\t" + dr.driver); // include rank
                outVal.set(dr.revenue);
                ctx.write(outKey, outVal);
                rank++;
            }
        }
    }

    @Override
    public int run(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage:\n  task1 [-r R] [-D filter.month=YYYY-MM] <input> <output>\n  task2 [-r R] <input> <jobA_output> <jobB_output>");
            return 2;
        }

        final String mode = args[0];
        int i = 1;
        int reducers = -1;

        // Simple option parse for -r
        while (i < args.length && args[i].startsWith("-")) {
            if ("-r".equals(args[i]) && (i + 1) < args.length) {
                reducers = Integer.parseInt(args[i + 1]);
                i += 2;
            } else {
                // let Hadoop conf options (-D ...) pass through
                i++;
            }
        }

        if ("task1".equalsIgnoreCase(mode)) {
            if (args.length - i != 2) {
                System.err.println("task1 expects <input> <output>");
                return 2;
            }
            String in = args[i];
            String out = args[i + 1];
            return runTask1(getConf(), in, out, reducers);

        } else if ("task2".equalsIgnoreCase(mode)) {
            if (args.length - i != 3) {
                System.err.println("task2 expects <input> <jobA_output> <jobB_output>");
                return 2;
            }
            String in = args[i];
            String mid = args[i + 1];
            String out = args[i + 2];
            return runTask2Pipeline(getConf(), in, mid, out, reducers);

        } else {
            System.err.println("Unknown mode: " + mode);
            return 2;
        }
    }

    private int runTask1(Configuration baseConf, String input, String output, int reducers) throws Exception {
        Configuration conf = new Configuration(baseConf);
        Job job = Job.getInstance(conf, "A3-Task1-DateRevenue");
        job.setJarByClass(Assignment3.class);

        job.setInputFormatClass(TextInputFormat.class);
        job.setMapperClass(DateRevenueMapper.class);
        job.setCombinerClass(SumDoubleCombiner.class); // safe for sum
        job.setReducerClass(SumDoubleReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        if (reducers >= 0) job.setNumReduceTasks(reducers);

        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));

        return job.waitForCompletion(true) ? 0 : 1;
    }

    private int runTask2Pipeline(Configuration baseConf, String input, String midOutput, String finalOutput, int reducers) throws Exception {
        // Job A: revenue per driver
        Configuration confA = new Configuration(baseConf);
        Job jobA = Job.getInstance(confA, "A3-Task2-DriverRevenue");
        jobA.setJarByClass(Assignment3.class);

        jobA.setInputFormatClass(TextInputFormat.class);
        jobA.setMapperClass(DriverRevenueMapper.class);
        // IMPORTANT (per assignment): do NOT blindly set combiner. We'll omit it.
        jobA.setReducerClass(DriverRevenueReducer.class);

        jobA.setMapOutputKeyClass(Text.class);
        jobA.setMapOutputValueClass(DoubleWritable.class);
        jobA.setOutputKeyClass(Text.class);
        jobA.setOutputValueClass(DoubleWritable.class);

        if (reducers >= 0) jobA.setNumReduceTasks(reducers);

        FileInputFormat.setInputPaths(jobA, new Path(input));
        FileOutputFormat.setOutputPath(jobA, new Path(midOutput));

        if (!jobA.waitForCompletion(true)) {
            return 1;
        }

        // Job B: top-10 from driver totals
        Configuration confB = new Configuration(baseConf);
        Job jobB = Job.getInstance(confB, "A3-Task2-Top10Drivers");
        jobB.setJarByClass(Assignment3.class);

        jobB.setInputFormatClass(TextInputFormat.class);
        jobB.setMapperClass(Top10Mapper.class);
        // Mapper emits key=1 for all -> force a single reducer to compute global top-10
        jobB.setMapOutputKeyClass(IntWritable.class);
        jobB.setMapOutputValueClass(Text.class);

        // SINGLE reducer required for global top-10
        jobB.setNumReduceTasks(1);
        jobB.setReducerClass(Top10Reducer.class);

        jobB.setOutputKeyClass(Text.class);
        jobB.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.setInputPaths(jobB, new Path(midOutput));
        FileOutputFormat.setOutputPath(jobB, new Path(finalOutput));

        return jobB.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new Assignment3(), args);
        System.exit(res);
    }
}

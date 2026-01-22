package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/hashicorp/go-cleanhttp"
	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

func main() {
	accessKeyID := os.Getenv("S3_ACCESS_KEY_ID")
	secretKey := os.Getenv("S3_ACCESS_KEY_SECRET_KEY")
	region := os.Getenv("S3_REGION")
	endpoint := os.Getenv("S3_ENDPOINT")
	gomaxprocs := os.Getenv("GOMAXPROCS")
	bucket := os.Getenv("S3_BUCKET")
	prefix := os.Getenv("S3_PREFIX")
	objectCount := os.Getenv("MAX_OBJECTS")
	testDuration := os.Getenv("TEST_DURATION")
	useSSL := os.Getenv("USE_SSL")
	readSizeKB := os.Getenv("READ_SIZE_KB")
	readBufferSizeKB := os.Getenv("READ_BUFFER_SIZE_KB")

	if accessKeyID == "" || secretKey == "" {
		slog.Error("S3_ACCESS_KEY_ID and S3_ACCESS_KEY_SECRET_KEY are required to run")

		os.Exit(1)
	}

	if region == "" || endpoint == "" {
		slog.Error("S3_REGION and S3_ENDPOINT are required to run")

		os.Exit(1)
	}

	if bucket == "" {
		slog.Error("S3_BUCKET is required to run")

		os.Exit(1)
	}

	threads := 1
	maxObjects := 1000
	var err error

	if gomaxprocs != "" {
		threads, err = strconv.Atoi(gomaxprocs)
		if err != nil {
			slog.Error("failed to convert GOMAXPROCS to int", "error", err)

			os.Exit(1)
		}
	}

	if objectCount != "" {
		maxObjects, err = strconv.Atoi(objectCount)
		if err != nil {
			slog.Error("failed to convert MAX_OBJECTS to int", "error", err)

			os.Exit(1)
		}
	}

	duration := time.Minute * 10

	if testDuration != "" {
		duration, err = time.ParseDuration(testDuration)
		if err != nil {
			slog.Error("failed to parse TEST_DURATION", "error", err)

			os.Exit(1)
		}
	}

	ssl := true
	if useSSL != "" {
		ssl, err = strconv.ParseBool(useSSL)
		if err != nil {
			slog.Error("failed to parse USE_SSL", "error", err)

			os.Exit(1)
		}
	}

	// 16KiB by default
	readSize := int64(16 * 1024)
	if readSizeKB != "" {
		val, err := strconv.Atoi(readSizeKB)
		if err != nil {
			slog.Error("failed to parse READ_SIZE_KB", "error", err)

			os.Exit(1)
		}

		readSize = int64(val * 1024)
	}

	// 32KiB by default
	readBufferSize := 32 * 1024
	if readBufferSizeKB != "" {
		val, err := strconv.Atoi(readBufferSizeKB)
		if err != nil {
			slog.Error("failed to parse READ_BUFFER_SIZE_KB", "error", err)

			os.Exit(1)
		}

		readBufferSize = val * 1024
	}

	ctx, cancel := context.WithCancel(context.Background())
	transport := cleanhttp.DefaultPooledTransport()

	// Modify some defaults to be more performant
	transport.MaxConnsPerHost = threads
	transport.DisableCompression = true

	// Our test only reads so just bump both buffers
	transport.ReadBufferSize = readBufferSize
	transport.WriteBufferSize = readBufferSize

	s3Client, err := minio.New(endpoint, &minio.Options{
		Creds:        credentials.NewStaticV4(accessKeyID, secretKey, ""),
		Secure:       ssl,
		Transport:    transport,
		Region:       region,
		BucketLookup: minio.BucketLookupDNS,
	})
	if err != nil {
		slog.Error("failed to create s3 client", "error", err)

		os.Exit(1)
	}

	var objects []string
	var sizes []int64

	objChan := s3Client.ListObjects(ctx, bucket, minio.ListObjectsOptions{
		WithMetadata: true,
		Prefix:       prefix,
		MaxKeys:      maxObjects,
	})

	for obj := range objChan {
		if obj.Err != nil {
			slog.Error("error listing objects", "error", err)

			os.Exit(1)
		}

		objects = append(objects, obj.Key)
		sizes = append(sizes, obj.Size)
	}

	rng := rand.New(rand.NewSource(time.Now().Unix()))
	timer := time.NewTimer(duration)
	resultChan := make(chan *testResult, threads)
	wg := &sync.WaitGroup{}

	params := &testParams{
		Bucket:      bucket,
		RNG:         rng,
		Objects:     objects,
		ObjectSizes: sizes,
		S3Client:    s3Client,
		BlockSize:   readSize,
	}

	start := time.Now().UTC()

	slog.Info("starting test")

	for i := range threads {
		wg.Add(1)
		go func() {
			result := runTest(ctx, params, i)
			resultChan <- result
			wg.Done()
		}()
	}

	<-timer.C

	slog.Info("test done")

	// Tell our tests that we're done and wait
	cancel()
	wg.Wait()

	aggregatedBytesRead := 0
	aggregatedRequestsSent := 0
	var (
		averageTTLB     int64
		averageTestTime int64
	)

	for range threads {
		result := <-resultChan

		aggregatedBytesRead += int(result.TotalBytesRead)
		aggregatedRequestsSent += int(result.TotalRequestsSent)
		averageTTLB += result.TotalTTLBMS
		averageTestTime += result.TotalTestTimeMS
	}

	averageTTLB = averageTTLB / int64(aggregatedRequestsSent)
	averageTestTime = averageTestTime / int64(aggregatedRequestsSent)

	timeSpent := time.Since(start)

	t := table.NewWriter()
	t.SetOutputMirror(os.Stdout)
	t.AppendHeader(table.Row{"", "RESULTS"})
	t.AppendRow(table.Row{"Time Spent", timeSpent.String()})
	t.AppendRow(table.Row{"Total Bytes Read", fmt.Sprintf("%d", aggregatedBytesRead)})
	t.AppendRow(table.Row{"Total Requests Sent", fmt.Sprintf("%d", aggregatedRequestsSent)})
	t.AppendRow(table.Row{"Average TTLB", fmt.Sprintf("%d", averageTTLB)})
	t.AppendRow(table.Row{"Average Test Time", fmt.Sprintf("%d", averageTestTime)})

	t.Render()
}

type testParams struct {
	Bucket      string
	BlockSize   int64
	RNG         *rand.Rand
	Objects     []string
	ObjectSizes []int64
	S3Client    *minio.Client
}

type testResult struct {
	TotalBytesRead    int64
	TotalRequestsSent int64
	TotalTestTimeMS   int64
	TotalTTLBMS       int64
}

func runTest(ctx context.Context, params *testParams, id int) *testResult {
	ll := slog.With("ID", id)

	result := &testResult{}

	testCtx := context.Background()

	for {
		select {
		case <-ctx.Done():
			return result
		default:
			testStart := time.Now().UTC()
			// Get our random object
			randObjIndex := params.RNG.Int() % len(params.Objects)
			obj := params.Objects[randObjIndex]
			size := params.ObjectSizes[randObjIndex]

			// Get a random 16KiB offset to read
			maxOffset := size / params.BlockSize
			randObjOffset := params.RNG.Int() % int(maxOffset)

			rangeStart := int64(randObjOffset) * params.BlockSize
			rangeEnd := min((rangeStart + params.BlockSize), size)

			result.TotalRequestsSent++

			reqStart := time.Now().UTC()

			reqOpts := minio.GetObjectOptions{}
			reqOpts.SetRange(rangeStart, rangeEnd)

			resp, err := params.S3Client.GetObject(testCtx, params.Bucket, obj, reqOpts)

			if err != nil {
				ll.Error("failed to fetch range", "error", err)

				continue
			}

			amount, err := io.Copy(io.Discard, resp)
			resp.Close()

			result.TotalBytesRead += amount
			result.TotalTTLBMS += time.Since(reqStart).Milliseconds()

			result.TotalTestTimeMS += time.Since(testStart).Milliseconds()

			if err != nil {
				ll.Error("failed to discard response body", "error", err)

				continue
			}
		}
	}
}

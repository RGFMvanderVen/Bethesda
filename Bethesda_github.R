
# =========================================
# LOAD DATASET
# ========================================= 

# Install packages
install.packages("readxl")
install.packages("ggplot2")

# Load dataset
library(readxl)
thyroid <- read_excel("Cmypath.xlsx")

# =========================================
# DATA SELECTION
# ========================================= 

# Create network variable:
thyroid1 <- thyroid
thyroid1$region <- ifelse (thyroid$hosp == 'other', 'rest van NL', 'network')
thyroid1$region

#Sort on Bethesda scores
thyroid1 <- thyroid1[order(thyroid1$Bethesda_combined),]

#Number of unique patients in the set: 
test <- thyroid1 %>%
  rename(ptnr = PALGApatiëntnr...1)
head(test)
length(unique(test$ptnr))

#Number of FNAs per patient:
observatie_count <- thyroid1 %>%
  group_by(`PALGApatiëntnr...1`) %>%
  summarise(aantal_observaties = n())

#Group by the number of observations and count how many people have that number of observations.
summary_count <- observatie_count %>%
  group_by(aantal_observaties) %>%
  summarise(aantal_personen = n())

print(summary_count)

#Counting number of observations per patient
#The entire country: 
patient_counts <- test %>%
  group_by(ptnr) %>%
  summarise(count = n())

min_count<- min(patient_counts$count)
max_count<- max(patient_counts$count)

median_count <- median(patient_counts$count)
iqr_count <- IQR(patient_counts$count)

cat("Minimal number of biopsies per patient:", min_count, "\n")
cat("Maximal number of biopsies per patient:", max_count, "\n")
cat("Median number of biopsies per patient:", median_count, "\n")
cat("IQR of number of biopsies per patient:", iqr_count, "\n")

#Rest_van_NL: 
test_restvanNL <- subset(test, region =="rest van NL")
patient_counts <- test_restvanNL %>%
  group_by(ptnr) %>%
  summarise(count = n())

min_count<- min(patient_counts$count)
max_count<- max(patient_counts$count)

median_count <- median(patient_counts$count)
iqr_count <- IQR(patient_counts$count)

cat("Minimal number of biopsies per patient:", min_count, "\n")
cat("Maximal number of biopsies per patient:", max_count, "\n")
cat("Median number of biopsies per patient:", median_count, "\n")
cat("IQR of number of biopsies per patient:", iqr_count, "\n")

#Network: 
test_network <- subset(test, region !="rest van NL")
patient_counts <- test_network %>%
  group_by(ptnr) %>%
  summarise(count = n())

min_count<- min(patient_counts$count)
max_count<- max(patient_counts$count)

median_count <- median(patient_counts$count)
iqr_count <- IQR(patient_counts$count)

cat("Minimal number of biopsies per patient:", min_count, "\n")
cat("Maximal number of biopsies per patient:", max_count, "\n")
cat("Median number of biopsies per patient:", median_count, "\n")
cat("IQR of number of biopsies per patient:", iqr_count, "\n")

#=====================================

# Subset maken met basis R subset()
subset_data <- subset(rnthyroid1, region == 'network')

# =====================================
# DISTRIBUTION OF BETHESDA SCORES 2010 - 2021
# =====================================

#Distribution overall Bethesda scores in the entire country: 
freq_table <- table(thyroid1$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

#Distribution overall Bethesda scores in the rest of the country (not network): 
sk4_rest <- subset(thyroid1, region=="rest van NL")
freq_table <- table(sk4_rest$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

#Distribution overall Bethesda scores in the network: 
sk4_nw <- subset(thyroid1, region=="network")
freq_table <- table(sk4_nw$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

# =====================================
# DISTRIBUTION OF BETHESDA SCORES 2010 - 2016
# =====================================

#Distribution 2010-2016 Bethesda scores in the entire country:
sk4_1016 <- subset(thyroid1, `Jaar onderzoek` <2017)
freq_table <- table(sk4_1016$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

#Distribution 2010-2016 Bethesda scores in the network
sk4_1016_nw <- subset(sk4_1016, region == 'network')
freq_table <- table(sk4_1016_nw$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

#Distribution 2010-2016 Bethesda scores rest of the country:
sk4_1016_rvNL <- subset(sk4_1016, region =='rest van NL')
freq_table <- table(sk4_1016_rvNL$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

# =====================================
# DISTRIBUTION BETHESDA SCORES 2017 - 2021
# =====================================

#Distribution 2017-2021 Bethesda scores in the entire country:
sk4_1721 <- subset(thyroid1, `Jaar onderzoek` >2016)
freq_table <- table(sk4_1721$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

#Distribution 2017-2021 Bethesda scores in the network:
sk4_1721_nw <- subset(sk4_1721, region == 'network')
freq_table <- table(sk4_1721_nw$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

#Distribution 2017-2021 Bethesda scores in the rest of the country:
sk4_1721_rvNL <- subset(sk4_1721, region == 'rest van NL')
freq_table <- table(sk4_1721_rvNL$Bethesda_combined)
print(freq_table)
percentages <- round(prop.table(freq_table)*100,1)
print(percentages)

# =========================================
# UNIVARIATE ANALYSIS
# ========================================= 

contingency_table <- table (thyroid1$Bethesda_combined, thyroid1$region)
chi_square_test <- chisq.test(contingency_table)
print(chi_square_test)

contingency_table2 <- table (sk4_1016$Bethesda_combined, sk4_1016$region)
chi_square_test2 <- chisq.test(contingency_table2)
print(chi_square_test2)

contingency_table3 <- table (sk4_1721$Bethesda_combined, sk4_1721$region)
chi_square_test3 <- chisq.test(contingency_table3)
print(chi_square_test3)

# =========================================
# PLOTING NUMBER OF FNAs OVER TIME
# ========================================= 
#For the entire country

heel_NL <- thyroid1

jaar_count <- table(heel_NL$`Jaar onderzoek`)
tellingen<-as.numeric(jaar_count)
tellingen
jaren<-as.numeric(names(jaar_count))
jaren
print(jaar_count)

plot(names(jaar_count), jaar_count, type="o", xlab="Jaar", ylab="Aantal", main="Trend in total number of FNAs in the Netherlands")
axis(1, at = jaren, labels = jaren)
axis(2, at = seq(0, max(tellingen)+1000, by = 1000))

png("C:/mypath/entirecountry.png")
plot(names(jaar_count), jaar_count, type="o", xlab="Jaar", ylab="Aantal", main="Trend in total number of FNAs in the Netherlands")
axis(1, at = jaren, labels = jaren)
axis(2, at = seq(0, max(tellingen)+1000, by = 1000))
dev.off()

#The rest of the country 
rest_van_NL <- subset(thyroid1, region =='rest van NL')

jaar_count <- table(rest_van_NL$`Jaar onderzoek`)
jaren<-as.numeric(names(jaar_count))
tellingen<-as.numeric(jaar_count)
print(jaar_count)

plot(names(jaar_count), jaar_count, type="o", xlab="Jaar", ylab="Aantal", main="Trend in total number of FNAs in the rest of the country (i.e. not the network)")
axis(1, at = jaren, labels = jaren)
axis(2, at = seq(0, max(tellingen)+1000, by = 1000))

png("C:/mypath/restofthecountry.png")
plot(names(jaar_count), jaar_count, type="o", xlab="Jaar", ylab="Aantal", main="Trend in total number of FNAs in the rest of the country (i.e. not the network)")
axis(1, at = jaren, labels = jaren)
axis(2, at = seq(0, max(tellingen)+1000, by = 1000))
dev.off()

#The network
network <- subset(thyroid1, region == 'network')

jaar_count <- table(network$`Jaar onderzoek`)
jaren<-as.numeric(names(jaar_count))
tellingen<-as.numeric(jaar_count)
print(jaar_count)

plot(names(jaar_count), jaar_count, type="o", xlab="Jaar", ylab="Aantal", main="Trend in total number of FNAs in the network")
axis(1, at = jaren, labels = jaren)
axis(2, at = seq(0, max(tellingen)+1000, by = 100))

png("C:/mypath/network.png")
plot(names(jaar_count), jaar_count, type="o", xlab="Jaar", ylab="Aantal", main="Trend in total number of FNAs in the network")
axis(1, at = jaren, labels = jaren)
axis(2, at = seq(0, max(tellingen)+1000, by = 100))
dev.off()

# =========================================
# MISSING DATA OVER TIME
# ========================================= 

# Checking the percentage of cases where the Bethesda score is mentioned in the conclusion and/or diagnosis text:

# Entire country
Bethesda_freq_heelNL <- heel_NL %>%
  count(Bethesda, .drop = FALSE) %>%
  mutate(percentage = n / sum(n)*100)
print(Bethesda_freq_heelNL)  

# Rest of the country
Bethesda_freq_restvanNL <- rest_van_NL %>%
  count(Bethesda, .drop = FALSE) %>%
  mutate(percentage = n / sum(n)*100)
print(Bethesda_freq_restvanNL)  

# Network
Bethesda_freq_network <- network %>%
  count(Bethesda, .drop = FALSE) %>%
  mutate(percentage = n / sum(n)*100)
print(Bethesda_freq_network)  

# Plotting the missing data over time:

# Entire country:

library(ggplot2)

missing_values <- heel_NL %>%
  summarise(total_count = n(), missing_count = sum(is.na(Bethesda))) %>%
  mutate(percentage_missing = (missing_count / total_count) * 100)
missing_values

# Entire country: Count of missing values per year
missing_values_per_year <- heel_NL %>%
  group_by(`Jaar onderzoek`) %>%
  summarise(total_count = n(), missing_count = sum(is.na(Bethesda))) %>%
  mutate(percentage_missing = (missing_count / total_count) * 100)
missing_values_per_year

# Entire country: Count of complete values per year
complete_data_per_year <- heel_NL %>%
  group_by(`Jaar onderzoek`) %>%
  summarise(total_count = n(), complete_count = sum(!is.na(Bethesda))) %>%
  mutate(percentage_complete = (complete_count / total_count) * 100)
complete_data_per_year

# Entire country: line chart
ggplot() +
  geom_line(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_line(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  geom_point(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_point(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  scale_color_manual(values = c("Missing" = "red", "Present in data" = "blue")) +
  labs(x = "Year of FNA", y = "Percentage", title = "Percentage missing vs. non-missing data over time")

png("C:/mypath/missingdatanationwide.png")
ggplot() +
  geom_line(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_line(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  geom_point(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_point(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  scale_color_manual(values = c("Missing" = "red", "Present in data" = "blue")) +
  labs(x = "Year of FNA", y = "Percentage", title = "Percentage missing vs. non-missing data over time")
dev.off()


# Rest of the country

# Rest of the country: Count of missing values per year
missing_values_per_year <- rest_van_NL %>%
  group_by(`Jaar onderzoek`) %>%
  summarise(total_count = n(), missing_count = sum(is.na(Bethesda))) %>%
  mutate(percentage_missing = (missing_count / total_count) * 100)
missing_values_per_year

# Rest of the country: Count van complete values per year
complete_data_per_year <- rest_van_NL %>%
  group_by(`Jaar onderzoek`) %>%
  summarise(total_count = n(), complete_count = sum(!is.na(Bethesda))) %>%
  mutate(percentage_complete = (complete_count / total_count) * 100)
complete_data_per_year


# Rest of the country: line chart
ggplot() +
  geom_line(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_line(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  geom_point(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_point(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  scale_color_manual(values = c("Missing" = "red", "Present in data" = "blue")) +
  labs(x = "Year of FNA", y = "Percentage", title = "Percentage missing vs. non-missing data over time")

png("C:/mypath/missingdatarestofthecountry.png")
ggplot() +
  geom_line(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_line(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  geom_point(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_point(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  scale_color_manual(values = c("Missing" = "red", "Present in data" = "blue")) +
  labs(x = "Year of FNA", y = "Percentage", title = "Percentage missing vs. non-missing data over time")
dev.off()

# The network

# Network: Count of missing values per year
missing_values <- network %>%
  summarise(total_count = n(), missing_count = sum(is.na(Bethesda))) %>%
  mutate(percentage_missing = (missing_count / total_count) * 100)
missing_values

missing_values_per_year <- network %>%
  group_by(`Jaar onderzoek`) %>%
  summarise(total_count = n(), missing_count = sum(is.na(Bethesda))) %>%
  mutate(percentage_missing = (missing_count / total_count) * 100)
missing_values_per_year

# Network: Count of complete values per year
complete_data_per_year <- network %>%
  group_by(`Jaar onderzoek`) %>%
  summarise(total_count = n(), complete_count = sum(!is.na(Bethesda))) %>%
  mutate(percentage_complete = (complete_count / total_count) * 100)
complete_data_per_year

# Network: line chart
ggplot() +
  geom_line(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_line(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  geom_point(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_point(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  scale_color_manual(values = c("Missing" = "red", "Present in data" = "blue")) +
  labs(x = "Year of FNA", y = "Percentage", title = "Percentage missing vs. non-missing data over time")

png("C:/Users/mypath/missingdatanetwork.png")
ggplot() +
  geom_line(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_line(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  geom_point(data = missing_values_per_year, aes(x = `Jaar onderzoek`, y = percentage_missing, color = "Missing")) +
  geom_point(data = complete_data_per_year, aes(x = `Jaar onderzoek`, y = percentage_complete, color = "Present in data")) +
  scale_color_manual(values = c("Missing" = "red", "Present in data" = "blue")) +
  labs(x = "Year of FNA", y = "Percentage", title = "Percentage missing vs. non-missing data over time")
dev.off()


# =========================================
# DISTRIBUTION BETHESDA SCORES - HEEL NEDERLAND
# ========================================= 

# Loop:
naamset <- heel_NL
hierkomtdetitel <- "Nationwide Trends in Bethesda Scores per Year"
titelpng <- "C:/mypath.png"

# Calculate the counts per category of Bethesda_combined per year
count_per_jaar_en_categorie <- table(naamset$`Jaar onderzoek`, naamset$Bethesda_combined)

# Determine the maximum of the counts
max_counts <- max(sapply(count_per_jaar_en_categorie[,1:6],max))

# maximize space for plot
par(mar=c(6, 5, 8, 8), xpd=TRUE)

# Create plot
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)

# Looping for every category:
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}

# Update axes
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)

# Add legend:
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)

png(titelpng)
par(mar=c(6, 5, 8, 8), xpd=TRUE)
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)
dev.off()


# =========================================
# DISTRIBUTION OF BETHESDA SCORES - REST OF THE COUNTRY
# ========================================= 

# Define the input for this specific visualization
naamset <- rest_van_NL
hierkomtdetitel <- "Rest of the country - Bethesda Score trends"
titelpng <- "C:/mypath.png"

count_per_jaar_en_categorie <- table(naamset$`Jaar onderzoek`, naamset$Bethesda_combined)
print(count_per_jaar_en_categorie)

max_counts <- max(sapply(count_per_jaar_en_categorie[,1:6],max))

par(mar=c(6, 5, 8, 8), xpd=TRUE)

# Create plot
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)

# Looping for each category
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}

axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)

# Add legend:
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)

png(titelpng)
par(mar=c(6, 5, 8, 8), xpd=TRUE)
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)
dev.off()

# =========================================
# DISTRIBUTION OF BETHESDA SCORES - NETWORK 
# ========================================= 

naamset <- network
hierkomtdetitel <- "Network Trends in Bethesda Scores per Year"
titelpng <- "C:/mypath.png"

# Calculate the counts per category of Bethesda_combined per year
count_per_jaar_en_categorie <- table(naamset$`Jaar onderzoek`, naamset$Bethesda_combined)
print(count_per_jaar_en_categorie)

max_counts <- max(sapply(count_per_jaar_en_categorie[,1:6],max))

par(mar=c(6, 5, 8, 8), xpd=TRUE)

# Create plot
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)

# Loop for all categories:
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}

# Update axes
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)

# Add legend
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)

png(titelpng)
par(mar=c(6, 5, 8, 8), xpd=TRUE)
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)
dev.off()


# =========================================
# DISTRIBUTION BETHESDA SCORES - HIGH-VOLUME SURGICAL CENTERS VS NON-HIGH-VOLUME SURGICAL CENTERS
# ========================================= 

network_highvolume <- subset(network, hosp %in% c( "hospA", "hospB", "hospC"))
network_nonhighvolume <- subset(network, !(hosp %in% c("hospA", "hospB", "hospC")))

# HIGH-VOLUME SURGICAL CENTERS 

# Define the input for this specific visualization
naamset <- network_highvolume
hierkomtdetitel <- "Network Trends in Bethesda Scores per Year - High-volume-surgical Centers"
titelpng <- "C:/mypath.png"

# Calculate the counts per category of Bethesda_combined per year.
count_per_jaar_en_categorie <- table(naamset$`Jaar onderzoek`, naamset$Bethesda_combined)
print(count_per_jaar_en_categorie)

max_counts <- max(sapply(count_per_jaar_en_categorie[,1:6],max))

par(mar=c(6, 5, 8, 8), xpd=TRUE)

# Creating plot
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)

# Loop for all categories:
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}

axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)

# Add legend:
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)

png(titelpng)
par(mar=c(6, 5, 8, 8), xpd=TRUE)
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)
dev.off()


# NON-HIGH-VOLUME SURGICAL CENTERS

# Define the input for this specific visualization
naamset <- network_nonhighvolume
hierkomtdetitel <- "Network Trends in Bethesda Scores per Year - Non-high-volume surgical centers"
titelpng <- "C:/mypath.png"

#Calculate the counts per category of Bethesda_combined per year.
count_per_jaar_en_categorie <- table(naamset$`Jaar onderzoek`, naamset$Bethesda_combined)
print(count_per_jaar_en_categorie)

max_counts <- max(sapply(count_per_jaar_en_categorie[,1:6],max))

par(mar=c(6, 5, 8, 8), xpd=TRUE)

# Create plot
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)

# Loop for all categories:
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}

axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)

# Add legend
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)

png(titelpng)
par(mar=c(6, 5, 8, 8), xpd=TRUE)
plot(rownames(count_per_jaar_en_categorie), count_per_jaar_en_categorie[,1], type = "o", col = 1, xlab = "Year", ylab = "Number of FNA", main = hierkomtdetitel, adj = 0.4, ylim=c(0, max_counts+500), cex.axis=0.8)
for (i in 1:6) {
  categorie <- unique(naamset$Bethesda_combined)[i]
  counts <- count_per_jaar_en_categorie[, as.character(categorie)]
  lines(jaren, counts, type ="o", col = i, pch = i)
}
axis(1, at = seq(2010, 2021, by = 1), labels = FALSE)
axis(2, at = seq(0, max(count_per_jaar_en_categorie), by = 1000), labels=FALSE)
legend("topright", legend = levels(naamset$Bethesda_combined),col = 1:6, pch = 1:6,title = "Bethesda Score", title.adj = c(6),xpd=TRUE, bty="n", inset = c(-0.3, 0.0), cex=.8)
dev.off()


# =========================================
# JOIN-POINT ANALYSIS
# ========================================= 

library(dplyr)
library(segmented)
library(ggplot2)

#====== DATA SET PREP ======;

# Filter the data for Bethesda score 1 and aggregate.
data_bs1 <- joint_data %>%
  filter(BS == 1) %>%
  group_by(jaar, region) %>%
  summarise(aantal_puncties = n(), .groups = 'drop')

#Same for B2
data_bs2 <- joint_data %>%
  filter(BS == 2) %>%
  group_by(jaar, region) %>%
  summarise(aantal_puncties = n(), .groups = 'drop')

#Same for B3
data_bs3 <- joint_data %>%
  filter(BS == 3) %>%
  group_by(jaar, region) %>%
  summarise(aantal_puncties = n(), .groups = 'drop')

#Same for B4
data_bs4 <- joint_data %>%
  filter(BS == 4) %>%
  group_by(jaar, region) %>%
  summarise(aantal_puncties = n(), .groups = 'drop')

#Same for B5
data_bs5 <- joint_data %>%
  filter(BS == 5) %>%
  group_by(jaar, region) %>%
  summarise(aantal_puncties = n(), .groups = 'drop')

#Same for B6
data_bs6 <- joint_data %>%
  filter(BS == 6) %>%
  group_by(jaar, region) %>%
  summarise(aantal_puncties = n(), .groups = 'drop')

# Split the data into 'network' and 'rest of the country'.
data_netwerk1 <- filter(data_bs1, region == "network")
data_rest1 <- filter(data_bs1, region == "rest van NL")

data_netwerk2 <- filter(data_bs2, region == "network")
data_rest2 <- filter(data_bs2, region == "rest van NL")

data_netwerk3 <- filter(data_bs3, region == "network")
data_rest3 <- filter(data_bs3, region == "rest van NL")

data_netwerk4 <- filter(data_bs4, region == "network")
data_rest4 <- filter(data_bs4, region == "rest van NL")

data_netwerk5 <- filter(data_bs5, region == "network")
data_rest5 <- filter(data_bs5, region == "rest van NL")

data_netwerk6 <- filter(data_bs6, region == "network")
data_rest6 <- filter(data_bs6, region == "rest van NL")

# ====== BETHESDA 1 SCORE ======;
# Defining the model
fit.glm_netwerk1<-glm(aantal_puncties~jaar, data=data_netwerk1)
summary(fit.glm_netwerk1)
fit.glm_rest1<-glm(aantal_puncties~jaar, data=data_rest1)
summary(fit.glm_rest1)

# Estimating breakpoints automatically
fit.seg_netwerk1 <- segmented(fit.glm_netwerk1, seg.Z = ~jaar)
summary(fit.seg_netwerk1)
fit.seg_rest1 <- segmented(fit.glm_rest1, seg.Z = ~jaar)
summary(fit.seg_rest1)

#Fitting the model
fit.seg_netwerk1<-segmented(fit.glm_netwerk1,seg.Z=~jaar, psi=2016)
fit.seg_rest1<-segmented(fit.glm_rest1,seg.Z = ~jaar, psi=2018)

# Summary and Davies tests
summary(fit.seg_netwerk1)
davies.test(fit.glm_netwerk1,~jaar, k=10) 
slope(fit.seg_netwerk1)

summary(fit.seg_rest1)
davies.test(fit.glm_rest1, ~jaar, k=10)
slope(fit.seg_rest1)

# Plotting the results
ggplot(data_bs1, aes(x = jaar, y = aantal_puncties, color = region)) +
  geom_point() +
  geom_line(data = data_netwerk1, aes(y = fitted(fit.seg_netwerk1))) +
  geom_line(data = data_rest1, aes(y = fitted(fit.seg_rest1))) +
  labs(title = "Joinpoint Trendanalyse Bethesda 1", 
       x = "Year", y = "Number of FNAs", color = "regionn") +
  theme_minimal()

# ====== BETHESDA 2 SCORE ======;
# Defining the model
fit.glm_netwerk2<-glm(aantal_puncties~jaar, data=data_netwerk2)
summary(fit.glm_netwerk2)
fit.glm_rest2<-glm(aantal_puncties~jaar, data=data_rest2)
summary(fit.glm_rest2)

# Estimating breakpoints automatically
fit.seg_netwerk2<-segmented(fit.glm_netwerk2, seg.Z=~jaar)
summary(fit.seg_netwerk2)
fit.seg_rest2<-segmented(fit.glm_rest2, seg.Z=~jaar)
summary(fit.seg_rest2)

# Fitting the model
fit.seg_netwerk2<-segmented(fit.glm_netwerk2,seg.Z=~jaar, npsi=1)
fit.seg_rest2<-segmented(fit.glm_rest2,seg.Z=~jaar, npsi=1)

# Summary and Davies tests
summary(fit.seg_netwerk2)
davies.test(fit.glm_netwerk2,~jaar, k=10)
slope(fit.seg_netwerk2)

summary(fit.seg_rest2)
davies.test(fit.glm_rest2, ~jaar, k=10)
slope(fit.seg_rest2)

# Plotting results
ggplot(data_bs2, aes(x = jaar, y = aantal_puncties, color = region)) +
  geom_point() +
  geom_line(data = data_netwerk2, aes(y = fitted(fit.seg_netwerk2))) +
  geom_line(data = data_rest2, aes(y = fitted(fit.seg_rest2))) +
  labs(title = "Joinpoint Trendanalyse Bethesda 2", 
       x = "Year", y = "Number of FNAs", color = "regionn") +
  theme_minimal()

# ====== BETHESDA 3 SCORE ======;
# Defining the model 
fit.glm_netwerk3<-glm(aantal_puncties~jaar, data=data_netwerk3)
summary(fit.glm_netwerk3)
fit.glm_rest3<-glm(aantal_puncties~jaar, data=data_rest3)
summary(fit.glm_rest3)

# Estimating breakpoints automatically
fit.seg_netwerk3<-segmented(fit.glm_netwerk3, seg.Z=~jaar)
summary(fit.seg_netwerk3)
fit.seg_rest3<-segmented(fit.glm_rest3, seg.Z=~jaar)
summary(fit.seg_rest3)

# Fitting the model 
fit.seg_netwerk3<-segmented(fit.glm_netwerk3,seg.Z=~jaar, psi=2019)
fit.seg_rest3<-segmented(fit.glm_rest3,seg.Z = ~jaar, psi=2016)

# Summary and Davies tests
summary(fit.seg_netwerk3)
davies.test(fit.glm_netwerk3,~jaar, k=10)
slope(fit.seg_netwerk3)

summary(fit.seg_rest3)
davies.test(fit.glm_rest3, ~jaar, k=10)
slope(fit.seg_rest3)

# Plotting the results
ggplot(data_bs3, aes(x = jaar, y = aantal_puncties, color = region)) +
  geom_point() +
  geom_line(data = data_netwerk3, aes(y = fitted(fit.seg_netwerk3))) +
  geom_line(data = data_rest3, aes(y = fitted(fit.seg_rest3))) +
  labs(title = "Joinpoint Trendanalyse Bethesda 3", 
       x = "Year", y = "Number of FNAs", color = "regionn") +
  theme_minimal()


# ===========================================================================================;
# Bonferroni correction for multiple testing
P_values<-c(9.802e-12, 2.588e-09,6.975e-15, 0.01, 0.03, 0.06)
p.adjust(P_values, method="bonferroni")

# ===========================================================================================;
# End script
# ===========================================================================================;

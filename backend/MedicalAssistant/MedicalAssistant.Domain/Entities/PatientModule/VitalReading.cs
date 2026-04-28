using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class VitalReading : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient? Patient { get; set; }

        public int ChronicDiseaseMonitorId { get; set; }
        public ChronicDiseaseMonitor? ChronicDiseaseMonitor { get; set; }

        public string ReadingType { get; set; } = string.Empty;
        public decimal Value { get; set; }
        public decimal? Value2 { get; set; }
        public string Unit { get; set; } = string.Empty;
        public string? SugarReadingContext { get; set; }
        public bool IsNormal { get; set; }
        public string RecordedBy { get; set; } = string.Empty;
        public string? Notes { get; set; }

        public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
    }
}

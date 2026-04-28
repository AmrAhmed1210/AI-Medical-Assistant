using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Domain.Entities.PatientModule
{
    public class VisitVitalSign : BaseEntity
    {
        public int PatientId { get; set; }
        public Patient Patient { get; set; } = null!;
        public int PatientVisitId { get; set; }
        public PatientVisit? PatientVisit { get; set; }

        public string Type { get; set; } = string.Empty;
        public decimal Value { get; set; }
        public decimal? Value2 { get; set; }
        public string Unit { get; set; } = string.Empty;
        public bool IsAbnormal { get; set; }
        public decimal? NormalRangeMin { get; set; }
        public decimal? NormalRangeMax { get; set; }
        public string RecordedBy { get; set; }    // doctor / nurse
        public string? Notes { get; set; }

        public DateTime RecordedAt { get; set; } = DateTime.UtcNow;
    }
}

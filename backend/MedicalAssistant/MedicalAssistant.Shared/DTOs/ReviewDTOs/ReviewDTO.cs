using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MedicalAssistant.Shared.DTOs.ReviewDTOs
{
    public class ReviewDto
    {
        public string Id { get; set; } = string.Empty;
        public string Author { get; set; } = string.Empty;
        public int Rating { get; set; }
        public string Comment { get; set; } = string.Empty;
        public DateTime Date { get; set; } // Use DateTime instead of string for proper date handling
    }
}